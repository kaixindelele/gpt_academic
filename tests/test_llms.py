# """
# 对各个llm模型进行单元测试
# """
def validate_path():
    import os, sys

    os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) + "/..")
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)


validate_path()  # validate path so you can run from base directory

if "在线模型":
    if __name__ == "__main__":
        # from request_llms.bridge_taichu import predict_no_ui_long_connection
        # from request_llms.bridge_cohere import predict_no_ui_long_connection
        # from request_llms.bridge_spark import predict_no_ui_long_connection
        # from request_llms.bridge_zhipu import predict_no_ui_long_connection
        # from request_llms.bridge_chatglm3 import predict_no_ui_long_connection
        from request_llms.bridge_google_gemini import predict_no_ui_long_connection, predict
        llm_kwargs = {
            "llm_model": 'gemini-1.5-flash-002',
            "max_length": 4096,
            "top_p": 1,
            'api_key': 'AIzaSyD3JPwlXFY6v_s8o5yV6dRnp0amunpcEHg',
            "temperature": 1,
        }

        result = predict_no_ui_long_connection(
            inputs="今天日期", llm_kwargs=llm_kwargs, history=[], sys_prompt="系统"
        )
        print("final result:", result)
        print("final result:", result)
