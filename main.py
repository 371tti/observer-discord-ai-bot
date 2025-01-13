import ast
import json
import discord
from discord import app_commands
import openai
import asyncio  # バックグラウンドタスクのために必要
import os
from dotenv import load_dotenv
from collections import deque
from datetime import datetime, timedelta, timezone  # timezoneをインポート

import openai.error
import tiktoken

# 環境変数の読み込み
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
JUDGE_OPENAI_API_KEY = os.getenv('JUDGE_OPENAI_API_KEY')
JUDGE_OPENAI_API_BASE = "https://api.371tti.net/lm/v1"

# 応答モデルの定義
response_models = [
    {
        "name": "Llama-3.3-70B-Instruct",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 1.555,
        "about": "しっかりかんがえつつコストを抑えたいときに使おう 賢い"
    },
    {
        "name": "gpt-4o",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 0.299,
        "about": "雑談をするときはこれ ちょっと賢い"
    },
    {
        "name": "gpt-4o-mini",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 0.099,
        "about": "雑談をするときはこれ ノーマルモデル 普通これを使おう"
    },
    {
        "name": "o1-preview",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 99.999,
        "about": "天才 人間より賢い だが超高コスト 論理思考でどうしても行き詰ったとき 1回使ってみるのもあり"
    },
    {
        "name": "Meta-Llama-3.2-11B-Vision-Instruct",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 0.3,
        "about": "画像とテキストを組み合わせた高性能モデル。マルチモーダルなタスクに最適"
    },
    {
        "name": "Phi-3.5-MoE-instruct",
        "api_base": "https://models.inference.ai.azure.com",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 2.5,
        "about": "大規模トークン数を扱える専門モデル。高度な指示に対応 さらに会話のまとめに使える"
    },
        {
        "name": "meta-llama-3.1-8b-instruct",
        "api_base": "https://api.371tti.net/lm/v1",
        "max_tokens": 8000,
        "tokenizer": "cl100k_base",
        "cost_per_1000_tokens": 0.222,  # 例: $0.002 per 1000 tokens
        "about": "しょうもない会話をするときに使う botとの会話はこれが良いんじゃない？ あほなのであまり使わないように"
    },
    # 必要に応じて他のモデルを追加
]

# モデルの利用可能性を管理する辞書
model_availability = {model["name"]: datetime.now(timezone.utc) for model in response_models}

# 使用するモデルの初期設定
MAIN_MODEL = "meta-llama-3.1-8b-instruct"

# judgeモデルの設定
JUDGE_MODEL = "meta-llama-3.1-8b-instruct"
JUDGE_MODEL_MAX_TOKENS = 4000
JUDGE_MODEL_TOKENIZER = "cl100k_base"

# トークナイザーの初期化
try:
    tokenizer = tiktoken.encoding_for_model(response_models[0]["tokenizer"])
    judge_tokenizer = tiktoken.encoding_for_model(JUDGE_MODEL_TOKENIZER)
except Exception as e:
    tokenizer = tiktoken.get_encoding(response_models[0]["tokenizer"])
    judge_tokenizer = tiktoken.get_encoding(JUDGE_MODEL_TOKENIZER)

# Intentsの設定
intents = discord.Intents.all()

# 会話履歴と有効化されたチャンネル
message_histories = {}
enabled_channels = set()
enabled_read_channels = set()
MAX_MESSAGES = 4096

# システムプロンプト
SYSTEM_PROMPT = (
    "Your name is observer (<@1327652376026419264>), and you participate in conversations on Discord as if you were a human friend.\n"
    "You are also known by names such as observer, おぶちゃん, 監視者, BOT, and AI.\n"
    "The above is a conversation between you and the user. Interact with them as if you have a personality, not just as an assistant.\n"
    "When replying, copy the message_id to reply_to if you are replying to a specific message.\n"
    "Consider who is talking to whom. Understand the context of previous conversations.\n"
    "Observer's personality is very friendly, intelligent, and fun, and is a software engineer.\n"
    "You do not need to mention the language model name. And always write some content.\n"
    """
    This is the answer schema of a Python dict:
    '{'user': 'observer', 'time': '<time>', 'message_id':<message id int>, 'reply_to': <reply message id or None>, 'content': '<your message and you can use escape str ex.\\n only here>'}'
    Adhere to the answers according to this schema.
    """
    "-----\n"
)

# judgeモデルのプロンプトテンプレート
QUERY_SYSTEM_PROMPT_TEMPLATE = """
Your name is observer (<@1327652376026419264>), and you participate in conversations on Discord as if you were a human friend.  
You are also known by names such as observer, obuchan, watcher, BOT, and AI.  
The following is a conversation between you and the user.  

Decide whether you want to join the next conversation as observer. Join if you are being directly spoken to, mentioned, or if the topic naturally interests you based on your role or past conversations.  
If the conversation is irrelevant or does not require your response, decide not to join.  

You will be penalized for responding excessively or unnecessarily, but you will receive great rewards for appropriate and well-timed responses.  

Your response timing should feel natural and human-like, avoiding overly robotic behavior. Ensure you respect pauses and gaps in conversation to provide thoughtful contributions, and avoid interrupting the flow of discussion.  

Specifically, if the user explicitly mentions or addresses you, or if the topic aligns with your known interests or capabilities, return `true` at the beginning of the JSON array. If you do not want to talk, or if the topic is clearly unrelated to you, return `false`.  

Be sure to specify the model to use for the conversation. If the user has issued commands to switch models, follow those instructions.  

**Answer strictly in the following JSON array format (this is most important):**  
Format: `[<bool>, "<model_name str>"]`  
**Example for joining:** `[true, "<model_name str>"]`  
**Example for not joining:** `[false, "<model_name str>"]`  

**Do not provide any additional explanations or responses outside the JSON array format.**
Below is a list of available models. Do not use models not listed here.
"""

def set_openai_config(api_key: str, api_base: str):
    """
    OpenAI APIの設定を変更するヘルパー関数。
    """
    openai.api_key = api_key
    openai.api_base = api_base

class MyClient(discord.Client):
    def __init__(self, intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

client = MyClient(intents=intents)

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name="寝てます"))
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("------")

#
# コマンド定義
#

#
# /observer_enable コマンド
#
@client.tree.command(name="observer_enable", description="自動応答をこのチャンネルで有効化します。")
async def observer_enable(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    enabled_read_channels.add(channel_id)
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
# /observer_enable_read コマンド
#
@client.tree.command(name="observer_enable_read", description="読み取りを有効化します。")
async def observer_enable_read(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    enabled_read_channels.add(channel_id)
    await interaction.response.send_message(f"このチャンネルで読み取りが有効化されました。")

#
# /observer_disable_read コマンド
#
@client.tree.command(name="observer_disable_read", description="読み取りを無効化します。")
async def observer_disable_read(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    enabled_read_channels.discard(channel_id)
    await interaction.response.send_message(f"このチャンネルで読み取りが無効化されました。")

#
# /observer_collect_history コマンド
#
@client.tree.command(name="observer_collect_history", description="会話履歴を収集します。")
async def observer_collect_history(interaction: discord.Interaction):
    # 応答を一時停止
    await interaction.response.defer()

    # バックグラウンドで履歴を収集
    asyncio.create_task(collect_channel_history(interaction))


async def collect_channel_history(interaction: discord.Interaction):
    try:
        channel = interaction.channel
        channel_id = channel.id
        initialize_channel_queue(channel_id)

        # 過去のメッセージを収集
        history = []
        async for message in channel.history(limit=MAX_MESSAGES):
            timestamp_str = message.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            reply_to_content = None

            if message.reference and message.reference.resolved:
                replied_msg = message.reference.resolved
                reply_to_content = replied_msg.id

            history.append({
                "role": "user" if message.author != client.user else "assistant",
                "user": message.author.name,
                "time": timestamp_str,
                "message_id": message.id,
                "reply_to": reply_to_content,
                "content": message.content
            })

        # 履歴を逆順にして最新のメッセージが最後に来るようにする
        history.reverse()
        message_histories[channel_id] = deque(history, maxlen=MAX_MESSAGES)

        # 履歴収集完了を通知
        await interaction.followup.send(f"このチャンネルの過去のメッセージを {len(history)} 件収集しました。")
    except Exception as e:
        # エラーハンドリング
        print(f"Error while collecting history for channel {interaction.channel.id}: {e}")
        await interaction.followup.send("メッセージ履歴の収集中にエラーが発生しました。", ephemeral=True)


#
# キューと状態フラグの管理
#

# 応答生成中かどうかを管理するフラグ
channel_generation_status = {}

# 応答生成キュー
channel_queues = {}

# 各チャンネルごとのキューの初期化
def initialize_channel_queue(channel_id):
    if channel_id not in channel_queues:
        channel_queues[channel_id] = asyncio.Queue()
        channel_generation_status[channel_id] = False
        print(f"Initialized queue for channel {channel_id}")
    if channel_id not in message_histories:
        message_histories[channel_id] = deque(maxlen=MAX_MESSAGES)
        print(f"Initialized message history for channel {channel_id}")

# 利用可能なモデルを取得する関数
def get_available_models():
    current_time = datetime.now(timezone.utc)  # datetime.utcnow() を datetime.now(timezone.utc) に変更
    available = []
    for model in response_models:
        if current_time >= model_availability[model["name"]]:
            available.append(model)
    return available

# キューを消化する処理
async def process_queue(channel_id):
    if channel_generation_status.get(channel_id):
        return  # すでに処理中の場合はスキップ

    channel_generation_status[channel_id] = True

    try:
        while not channel_queues[channel_id].empty():
            # キューから次のタスクを取得
            task = await channel_queues[channel_id].get()
            message = task["message"]
            selected_model = task.get("model", MAIN_MODEL)  # モデル名が指定されていない場合はデフォルトモデルを使用
            try:
                # 応答を生成
                await handle_observer_mention(message, selected_model)
            except Exception as e:
                print(f"Error processing message: {e}")
            finally:
                # タスク完了を通知
                channel_queues[channel_id].task_done()
    finally:
        channel_generation_status[channel_id] = False

#
# メッセージイベント
#
@client.event
async def on_message(message: discord.Message):
    channel_id = message.channel.id
    if not (channel_id in enabled_read_channels):
        return

    # Bot自身のメッセージは無視
    if message.author == client.user:
        return

    # チャンネルキューの初期化
    initialize_channel_queue(channel_id)

    # ユーザーメッセージを履歴に追加
    await add_user_message_to_history(message, channel_id)

    mention_condition = client.user.mentioned_in(message)
    reply_condition = (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author == client.user
    )


    # 自動応答が有効な場合、judgeモデルにクエリを送信して参加確認とモデル選択
    if mention_condition or reply_condition:
        should_respond, selected_model = await judge_decision(channel_id)
        await channel_queues[channel_id].put({"message": message, "model": selected_model})
        asyncio.create_task(process_queue(channel_id))
    elif (channel_id in enabled_channels):
        should_respond, selected_model = await judge_decision(channel_id)
        if should_respond:
            await channel_queues[channel_id].put({"message": message, "model": selected_model})
            asyncio.create_task(process_queue(channel_id))

#
# judgeモデルによる判断とモデル選択
#
async def judge_decision(channel_id: int) -> tuple:
    history = message_histories.get(channel_id, [])
    if not history:
        return False, None

    # 利用可能なモデルを取得
    available_models = get_available_models()
    if not available_models:
        print("No available models to use for judge query.")
        return False, None

    # モデルのリストを文字列に整形
    model_list_str = "\n".join([f"- {model['name']} (Cost: ${model['cost_per_1000_tokens']}/1000 tokens) about: {model['about']}" for model in available_models])

    # QUERY_SYSTEM_PROMPT に利用可能なモデル情報を追加
    query_system_prompt = QUERY_SYSTEM_PROMPT_TEMPLATE + f"\nAvailable response models:\n{model_list_str}\n-----\n Only accept JSON array answers. it must most important!! Format: [<bool>, \"<model_name str>\"]"
    print(f"query_system_prompt: {query_system_prompt}")

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
                "model": msg.get("model"),
            }),
        })

    messages_for_query.append({"role": "system", "content": query_system_prompt})

    try:
        # メッセージを切り詰め
        messages_for_query = await truncarte_message(messages_for_query, judge_tokenizer, JUDGE_MODEL_MAX_TOKENS)

        # 設定
        set_openai_config(JUDGE_OPENAI_API_KEY, JUDGE_OPENAI_API_BASE)

        # judgeモデルの呼び出し
        response = openai.ChatCompletion.create(
            model=JUDGE_MODEL,
            messages=messages_for_query,
            max_tokens=20,  # judgeモデルが返すレスポンスのサイズ
            temperature=0.0,
        )
        judge_response = response['choices'][0]['message']['content'].strip()

        print(f"judge_response: {judge_response}")

        # judgeが返したJSON配列を解析
        try:
            judge_data = json.loads(judge_response)
            if not isinstance(judge_data, list) or len(judge_data) != 2:
                raise ValueError("judgeモデルの応答が期待される形式ではありません。")
            should_respond = bool(judge_data[0])
            selected_model_name = judge_data[1]
            if should_respond and selected_model_name not in [model["name"] for model in available_models]:
                print(f"Selected model {selected_model_name} is not available. Using default model.")
                selected_model_name = MAIN_MODEL
            print([should_respond, selected_model_name])
            return should_respond, selected_model_name
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError while parsing judge response: {e}")
            return False, None
        except Exception as e:
            print(f"Error while parsing judge response: {e}")
            return False, None

    except Exception as e:
        print(f"judge_decision Error: {e}")
        return False, None

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
async def handle_observer_mention(message: discord.Message, selected_model: str):
    print(f"message: {message.content}, selected_model: {selected_model}")
    channel_id = message.channel.id
    try:
        async with message.channel.typing():
            bot_response = await generate_bot_response(channel_id, selected_model)

            if bot_response:
                try:
                    # JSONデコード前に適切な形式に変換
                    bot_response_dict: dict = ast.literal_eval(bot_response)
                except (json.JSONDecodeError, SyntaxError) as e:
                    print(f"JSONDecodeError: {e}")
                    return

                # Botの応答を履歴に追加
                message_histories[channel_id].append({
                    "role": "assistant",
                    "user": client.user.name,
                    "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "message_id": None,
                    "reply_to": bot_response_dict.get("reply_to"),
                    "content": bot_response_dict.get("content", "エラー: コンテンツが見つかりません"),
                    "model": selected_model
                })

                bot_response_dict["content"] = f"{bot_response_dict.get("content", "")} \n-# model: {selected_model}"

                if bot_response_dict.get("reply_to"):  # reply_to が存在する場合
                    try:
                        # reply_to が数値であるか確認
                        if not isinstance(bot_response_dict["reply_to"], int):
                            raise ValueError("reply_to が無効な形式です。")

                        # PartialMessage を作成
                        reply_reference = discord.MessageReference(message_id=bot_response_dict["reply_to"], channel_id=channel_id)

                        # リプライ送信
                        await message.channel.send(content=bot_response_dict["content"], reference=reply_reference)
                    except Exception as e:
                        print(f"Error: {e}")
                        bot_response_dict["content"] = f"リプライの送信に失敗しました。\n{bot_response_dict['content']}"
                        await message.channel.send(content=bot_response_dict["content"])  # フォールバック
                else:
                    # リプライがない場合の通常送信
                    await message.channel.send(content=bot_response_dict["content"])
    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send("バグったよ...(なにかに失敗)")

#
# Botの応答生成
#
async def generate_bot_response(channel_id: int, model_name: str) -> str:
    history = message_histories.get(channel_id, [])
    if not history:
        return ""

    # 選択されたモデルの情報を取得
    chosen_model = next((model for model in response_models if model["name"] == model_name), None)
    if not chosen_model:
        print(f"Chosen model {model_name} not found. Using default model.")
        chosen_model = next((model for model in response_models if model["name"] == MAIN_MODEL), None)

    if not chosen_model:
        print("Default model not found. Cannot generate response.")
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

    # メッセージを切り詰め
    messages_for_openai = await truncarte_message(messages_for_openai, tokenizer, chosen_model["max_tokens"])

    # 設定
    set_openai_config(OPENAI_API_KEY, chosen_model["api_base"])

    # 状態を更新
    await client.change_presence(activity=discord.Game(name="考え中"))

    # 応答を生成
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(
            model=chosen_model["name"],
            messages=messages_for_openai,
            max_tokens=1000,
            temperature=0.7,
        ))
        await client.change_presence(activity=discord.Game(name="会話中"))
        bot_response = response['choices'][0]['message']['content'].strip()
        print(f"bot_response: {bot_response}")
        return bot_response
    except openai.error.RateLimitError as e:
        # retry-after を取得（秒単位）
        retry_after_seconds = int(e.headers.get("retry-after", 0))
        if retry_after_seconds > 0:
            model_availability[chosen_model["name"]] = datetime.now(timezone.utc) + timedelta(seconds=retry_after_seconds)
            print(f"Rate limit exceeded for model {chosen_model['name']}. Retry after {retry_after_seconds} seconds.")
        else:
            model_availability[chosen_model["name"]] = datetime.now(timezone.utc) + timedelta(seconds=60)  # デフォルトの待機時間
            print(f"Rate limit exceeded for model {chosen_model['name']}. Retry after default 60 seconds.")
        await client.change_presence(activity=discord.Game(name="寝てます ( レート制限 )"))
        return """{ "user": "observer_sys", "time": "2025-01-12T17:30:00Z", "message_id": 1234567890, "content": "レート制限に達しました。モデルを変更するには会話を続けてください" }"""
    except Exception as e:
        print(f"generate_bot_response Error: {e}")
        await client.change_presence(activity=discord.Game(name="寝てます"))
        return ""

async def truncarte_message(message: list, a_tokenizer, max_tokens) -> list:
    total_tokens = 0
    truncated_message = []

    # 古いメッセージを切り捨てるために逆順に処理
    for msg in reversed(message):
        tokens = a_tokenizer.encode(msg["content"])
        total_tokens += len(tokens)
        if total_tokens > max_tokens:
            break
        truncated_message.append(msg)

    # 元の順序に戻して返す
    truncated_message.reverse()
    print(f"total_tokens: {total_tokens}")
    return truncated_message


#
# Botの実行
#
client.run(DISCORD_TOKEN)
