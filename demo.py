import sys
import random
sys.path.append("/gpt/gpt_academic")
from toolbox import get_conf
API_KEYS = get_conf("SILICONFLOW_API_KEYS")
print("API_KEYS:", API_KEYS)

api_keys = get_conf("SILICONFLOW_API_KEYS")
api_key = random.choice(api_keys.split(','))

print("api_key:", api_key)