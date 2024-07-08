import re
import os
import time
from functools import wraps, lru_cache
from shared_utils.advanced_markdown_format import format_io
from shared_utils.config_loader import get_conf as get_conf


pj = os.path.join
default_user_name = 'default_user'
CACHE_FILE = 'sql.txt'
CACHE_INTERVAL = 600  # 10分钟
import json
from get_api_sql import DBManager
db_manager = DBManager()

def get_db_data():
    apikey, url = None, None
    print("get_db_connection:", db_manager.server)

    for _ in range(3):
        apikey = db_manager.get_single_alive_key()
        url = db_manager.get_single_alive_key_url(apikey)
        if apikey and url:
            print("get_db_data:", apikey, url)
            break
    return {'apikey': apikey, 'url': url}

def read_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print("read cache error:", e)
            return None
    return None

def write_cache(data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f)

def get_data():
    cache = read_cache()
    current_time = time.time()

    if cache and current_time - cache['timestamp'] < CACHE_INTERVAL:
        print("Reading from cache", cache['data'])
        return cache['data']['apikey'], cache['data']['url']

    print("Fetching from database")
    data = get_db_data()
    print("data:", data)
    cache_data = {
        'timestamp': current_time,
        'data': data
    }
    if data['apikey'] and data['url']:
        write_cache(cache_data)
    return data['apikey'], data['url']

def is_openai_api_key(key):
    CUSTOM_API_KEY_PATTERN = get_conf('CUSTOM_API_KEY_PATTERN')
    if len(CUSTOM_API_KEY_PATTERN) != 0:
        API_MATCH_ORIGINAL = re.match(CUSTOM_API_KEY_PATTERN, key)
    else:
        API_MATCH_ORIGINAL = re.match(r"sk-[a-zA-Z0-9]{48}$|sk-proj-[a-zA-Z0-9]{48}$|sess-[a-zA-Z0-9]{40}$", key)
    return bool(API_MATCH_ORIGINAL)


def is_azure_api_key(key):
    API_MATCH_AZURE = re.match(r"[a-zA-Z0-9]{32}$", key)
    return bool(API_MATCH_AZURE)


def is_api2d_key(key):
    API_MATCH_API2D = re.match(r"fk[a-zA-Z0-9]{6}-[a-zA-Z0-9]{32}$", key)
    return bool(API_MATCH_API2D)


def is_cohere_api_key(key):
    API_MATCH_AZURE = re.match(r"[a-zA-Z0-9]{40}$", key)
    return bool(API_MATCH_AZURE)


def is_any_api_key(key):
    if ',' in key:
        keys = key.split(',')
        for k in keys:
            if is_any_api_key(k): return True
        return False
    else:
        return is_openai_api_key(key) or is_api2d_key(key) or is_azure_api_key(key) or is_cohere_api_key(key)


def what_keys(keys):
    avail_key_list = {'OpenAI Key': 0, "Azure Key": 0, "API2D Key": 0}
    key_list = keys.split(',')

    for k in key_list:
        if is_openai_api_key(k):
            avail_key_list['OpenAI Key'] += 1

    for k in key_list:
        if is_api2d_key(k):
            avail_key_list['API2D Key'] += 1

    for k in key_list:
        if is_azure_api_key(k):
            avail_key_list['Azure Key'] += 1

    return f"检测到： OpenAI Key {avail_key_list['OpenAI Key']} 个, Azure Key {avail_key_list['Azure Key']} 个, API2D Key {avail_key_list['API2D Key']} 个"


def select_api_key(keys, llm_model):
    import random
    avail_key_list = []
    key_list = keys.split(',')

    if llm_model.startswith('gpt-') or llm_model.startswith('one-api-'):
        for k in key_list:
            if is_openai_api_key(k): avail_key_list.append(k)

    if llm_model.startswith('api2d-'):
        for k in key_list:
            if is_api2d_key(k): avail_key_list.append(k)

    if llm_model.startswith('azure-'):
        for k in key_list:
            if is_azure_api_key(k): avail_key_list.append(k)

    if llm_model.startswith('cohere-'):
        for k in key_list:
            if is_cohere_api_key(k): avail_key_list.append(k)

    if len(avail_key_list) == 0:
        raise RuntimeError(f"您提供的api-key不满足要求，不包含任何可用于{llm_model}的api-key。您可能选择了错误的模型或请求源（左上角更换模型菜单中可切换openai,azure,claude,cohere等请求源）。")

    api_key = random.choice(avail_key_list) # 随机负载均衡
    return api_key


def select_api_key(keys, llm_model):
    import random
    avail_key_list = []
    keys = keys.strip()
    if "," in keys:
        key_list = keys.split(',')
    elif "\n" in keys:
        key_list = keys.split('\n')
    else:
        # print("keys:", keys)
        key_list = [keys]
    if len(key_list) > 1:
        key_list = [key for key in key_list if len(key) > 0]    

    print("key_list:", len(key_list))
    # 判断用户是不是VIP。
    try:
        with open('vip_apis.txt', 'r', encoding='utf8') as f:
            vip_apis = f.read()
            vip_apis = vip_apis.strip()
            if "\n" in vip_apis:
                vip_apis = vip_apis.split('\n')
            elif "," in vip_apis:
                vip_apis = vip_apis.split(',')
            elif len(vip_apis) == 51:
                vip_apis = [vip_apis]
        if len(vip_apis) > 0:
            key_list = vip_apis
    except Exception as e:
        # print("读取VIP名单失败，将不会使用VIP列表", e)
        vip_apis = []

    # 判断有没有Cohere的api。
    try:
        with open('cohere_apis.txt', 'r', encoding='utf8') as f:
            cohere_apis = f.read()
            cohere_apis = cohere_apis.strip()
            if "," in cohere_apis:
                cohere_apis = cohere_apis.split(',')
    except Exception as e:
        # print("读取Cohere名单失败，将不会使用Cohere列表", e)
        cohere_apis = []
    print("cohere_apis:", len(cohere_apis))
    print("vip_apis_num:", len(vip_apis))
    try:
        with open('black_apis.txt', 'r', encoding='utf8') as f:
            black_apis = f.read().split('\n')
    except Exception as e:
        print("读取黑名单失败，将不会使用黑名单", e)
        black_apis = []

    if llm_model.startswith('gpt-'):
        for k in key_list:
            # 在这儿判断这些key是否在黑名单中
            if is_openai_api_key(k) and k not in black_apis:
                avail_key_list.append(k)

    if llm_model.startswith('api2d-'):
        for k in key_list:
            if is_api2d_key(k): avail_key_list.append(k)

    if llm_model.startswith('azure-'):
        for k in key_list:
            if is_azure_api_key(k): avail_key_list.append(k)

    if llm_model.startswith('command-') or llm_model.startswith('cohere-') and len(cohere_apis) > 0:
        for k in cohere_apis:
            if is_cohere_api_key(k): avail_key_list.append(k)
        api_key = random.choice(avail_key_list) # 随机负载均衡
        print("current_selected_command_api_key:", api_key)
        return api_key

    if len(avail_key_list) == 0:
        raise RuntimeError(f"您提供的api-key不满足要求，不包含任何可用于{llm_model}的api-key。您可能选择了错误的模型或请求源（右下角更换模型菜单中可切换openai,azure,claude,api2d等请求源）。")
    print("live_apis_num:", len(avail_key_list))

    if llm_model.startswith('gpt-'):
        api_key = random.choice(avail_key_list)
        # # 从mysql数据库中选一个：
        # apikey, url = get_data()
        # print("mysql_apikey:", apikey)
        # print("mysql_apikey url:", url)

        # avail_key_list = [apikey]

        # api_key = random.choice(avail_key_list) # 随机负载均衡
        print("current_selected_api_key:", api_key)

    return api_key