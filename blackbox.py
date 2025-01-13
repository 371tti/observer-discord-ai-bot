import ast
import json
import discord
from discord import app_commands
import openai
import os
from dotenv import load_dotenv
from collections import deque
from datetime import datetime, timezone

import tiktoken

# 環境変数の読み込み
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_BASE = "https://models.inference.ai.azure.com" #"https://api.371tti.net/lm/v1"
MODEL = "gpt-4o-mini" #"meta-llama-3.1-8b-instruct"
MAX_TOKEN_LEN = 8000

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = OPENAI_API_BASE
tokenizer = tiktoken.encoding_for_model(MODEL)

# Intentsの設定
intents = discord.Intents.all()

# 会話履歴と有効化されたチャンネル
message_histories = {}
enabled_channels = set()
MAX_MESSAGES = 100

# システムプロンプト
SYSTEM_PROMPT = (
    "あなたはobserver (ID: @1327652376026419264) という名前で、discord上で人間の友達のように会話に参加します。\n"
    "以上はユーザーとあなたの会話です。 \n"
    "リプライ先がある場合、そのメッセージに合わせて自然に返答してください。\n"
    "あなたはメッセージをよみとって応答すべき内容を応答してください"
    "Answer Schema: '{'user': 'observer', 'time': '<time>', 'message_id':<message id int>, 'reply_to': <replay massage id or None>, 'content': '<your message>'}'"
    "-----\n"
)

QUERY_SYSTEM_PROMPT = (
    "あなたはobserver (ID: @1327652376026419264) という名前で、discord上で人間の友達のように会話に参加します。\n"
    "以上はユーザーとあなたの会話です。 \n"
    "次の会話にobserverとして参加するべきか判断してください。 "
    "具体的には あなたに対して話されている場合、あなたの興味のある話の場合はtrue, あなたが話したくない場合、またはあなたに関係のない話の場合はfalseを返してください。"
    "Should you join this conversation next?"
    "Only accept boolean answers.\n"
    "true: join, false: don't join"
    "-----\n"
)

class MyClient(discord.Client):
    def __init__(self, intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
    
    async def setup_hook(self):
        await self.tree.sync()

client = MyClient(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("------")

#
# /observer_enable コマンド
#
@client.tree.command(name="observer_enable", description="自動応答をこのチャンネルで有効化します。")
async def observer_enable(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    enabled_channels.add(channel_id)
    await interaction.response.send_message(f"このチャンネルで自動応答が有効化されました。")

#
# /observer_disable コマンド
#
@client.tree.command(name="observer_disable", description="自動応答をこのチャンネルで無効化します。")
async def observer_disable(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    enabled_channels.discard(channel_id)
    await interaction.response.send_message(f"このチャンネルで自動応答が無効化されました。")

#
# /observer_status コマンド
#
@client.tree.command(name="observer_status", description="自動応答の状態を表示します。")
async def observer_status(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    status = "有効" if channel_id in enabled_channels else "無効"
    await interaction.response.send_message(f"このチャンネルの自動応答状態: {status}")

#
# /observer_reset コマンド
#
@client.tree.command(name="observer_reset", description="会話履歴をリセットします。")
async def observer_reset(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    message_histories[channel_id] = deque(maxlen=MAX_MESSAGES)
    await interaction.response.send_message(f"このチャンネルの会話履歴をリセットしました。")

#
# /observer_load コマンド
#
@client.tree.command(name="observer_load", description="会話履歴を読み込みます。")
async def observer_load(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    history = message_histories.get(channel_id, [])
    await interaction.response.send_message(f"このチャンネルの会話履歴: {len(history)}件")

#
# メッセージイベント
#
@client.event
async def on_message(message: discord.Message):
    # Bot自身のメッセージは無視
    if message.author == client.user:
        return

    channel_id = message.channel.id

    if channel_id not in message_histories:
        message_histories[channel_id] = deque(maxlen=MAX_MESSAGES)

    # ユーザーメッセージを履歴に追加
    await add_user_message_to_history(message, channel_id)

    # メンションまたはリプライがBot宛てなら応答
    mention_condition = client.user.mentioned_in(message)
    reply_condition = (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author == client.user
    )

    # メンションやリプライで呼び出された場合
    if mention_condition or reply_condition:
        await handle_observer_mention(message)
    else:
        # 有効化されていないチャンネルでは応答しない
        if channel_id not in enabled_channels:
            return
        # 自動応答が有効な場合、AIにクエリを送信して参加確認
        should_respond = await query_ai_for_response(channel_id)
        if should_respond:
            await handle_observer_mention(message)

#
# AIに会話参加確認（bool query）
#
async def query_ai_for_response(channel_id: int) -> bool:
    history = message_histories.get(channel_id, [])
    if not history:
        return False

    # OpenAI API用のメッセージリストを生成
    messages_for_query = []

    for msg in history:
        messages_for_query.append({
            "role": msg["role"],
            "content": str({
                "user": msg["user"],
                "time": msg["time"],
                "message_id": msg["message_id"],
                "reply_to": msg.get("reply_to"),
                "content": msg["content"],
            }),
        })

    messages_for_query.append({"role": "system", "content": QUERY_SYSTEM_PROMPT})

    try:
        response = await generate(messages_for_query, 0.0)
        response = ast.literal_eval(response)
        bot_response = response['choices'][0]['text']['content'][:2]
        return not(bot_response in ["no", "fa", "NO", "FA", "Fa", "No"])
    except Exception as e:
        print(f"query_ai_for_response Error: {e}")
        return False

#
# ユーザーのメッセージを履歴に追加
#
async def add_user_message_to_history(message: discord.Message, channel_id: int):
    timestamp_str = message.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    reply_to_content = None

    if message.reference and message.reference.resolved:
        replied_msg = message.reference.resolved
        reply_to_content = replied_msg.id

    # JSONに新しい要素を追加
    message_histories[channel_id].append({
        "role": "user",
        "user": message.author.name,
        "time": timestamp_str,
        "message_id": message.id,
        "reply_to": reply_to_content,
        "content": message.content
    })

#
# メンションやリプライで呼び出された場合
#
#
# メンションやリプライで呼び出された場合
#
async def handle_observer_mention(message: discord.Message):
    channel_id = message.channel.id
    try:
        async with message.channel.typing():
            bot_response = await generate_bot_response(channel_id)

            if bot_response:
                try:
                    # JSONデコード前に適切な形式に変換
                    bot_response: dict = ast.literal_eval(bot_response)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    await message.channel.send("応答の解析に失敗しました。")
                    return

                # Botの応答を履歴に追加
                message_histories[channel_id].append({
                    "role": "assistant",
                    "user": client.user.name,
                    "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "message_id": None,
                    "reply_to": None,
                    "content": bot_response.get("content", "エラー: コンテンツが見つかりません")
                })

                if bot_response.get("reply_to"):  # reply_to が存在する場合
                    try:
                        # reply_to が数値であるか確認
                        if not isinstance(bot_response["reply_to"], int):
                            raise ValueError("reply_to が無効な形式です。")
                        
                        # PartialMessage を作成
                        reply_reference = discord.PartialMessage(channel=message.channel, id=bot_response["reply_to"])

                        # リプライ送信
                        await message.channel.send(content=bot_response["content"], reference=reply_reference)
                    except Exception as e:
                        print(f"Error: {e}")
                        await message.channel.send("リプライの送信に失敗しました。")
                        await message.channel.send(content=bot_response["content"])  # フォールバック
                else:
                    # リプライがない場合の通常送信
                    await message.channel.send(content=bot_response["content"])
    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send("バグったよ...(なにかに失敗)")


#
# Botの応答生成
#
async def generate_bot_response(channel_id: int) -> str:
    history = message_histories.get(channel_id, [])
    if not history:
        return ""

    messages_for_openai = []    

    for msg in history:
        messages_for_openai.append({
            "role": msg["role"],
            "content": str({
                "user": msg["user"],
                "time": msg["time"],
                "message_id": msg["message_id"],
                "reply_to": msg.get("reply_to"),
                "content": msg["content"],
            }),
        })

    # OpenAI API用のメッセージリストを生成
    messages_for_openai.append({"role": "system", "content": SYSTEM_PROMPT})

    try:
        response = await generate(messages_for_openai, 0.7)
        bot_response = response['choices'][0]['text']['content'].strip()
        return bot_response
    except Exception as e:
        print(f"generate_bot_response Error: {e}")
        return ""
    
async def truncarte_message(message: list) -> list:
    total_tokens = 0
    truncated_message = []
    for msg in message:
        tokens = tokenizer.encode(msg["content"])
        total_tokens += len(tokens)
        if total_tokens > MAX_TOKEN_LEN:
            break
        truncated_message.append(msg)
    print(f"total_tokens: {total_tokens}")
    print("truncated_message: ", truncated_message)
    return truncated_message
    
async def generate(message: list, temperature: float) -> dict:
    message = await truncarte_message(message)
    response = openai.Completion.create(
        model=MODEL,
        prompt=json.dumps(message),
        max_tokens=100,
        temperature=temperature,
    )
    print(f"response: {response}")
    return response

client.run(DISCORD_TOKEN)
