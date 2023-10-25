"""
对项目中的各个插件进行测试。运行方法：直接运行 python tests/test_plugins.py
"""


import os, sys
def validate_path(): dir_name = os.path.dirname(__file__); root_dir_assume = os.path.abspath(dir_name +  '/..'); os.chdir(root_dir_assume); sys.path.append(root_dir_assume)
validate_path() # 返回项目根路径

if __name__ == "__main__":
    from tests.test_utils import plugin_test
    # plugin_test(plugin='crazy_functions.函数动态生成->函数动态生成', main_input='交换图像的蓝色通道和红色通道', advanced_arg={"file_path_arg": "./build/ants.jpg"})

    plugin_test(plugin='crazy_functions.Latex输出PDF结果->Latex翻译中文并重新编译PDF', main_input="2307.07522")

    # plugin_test(plugin='crazy_functions.虚空终端->虚空终端', main_input='修改api-key为sk-jhoejriotherjep')

    # plugin_test(plugin='crazy_functions.批量翻译PDF文档_NOUGAT->批量翻译PDF文档', main_input='crazy_functions/test_project/pdf_and_word/aaai.pdf')

    # plugin_test(plugin='crazy_functions.虚空终端->虚空终端', main_input='调用插件，对C:/Users/fuqingxu/Desktop/旧文件/gpt/chatgpt_academic/crazy_functions/latex_fns中的python文件进行解析')

    # plugin_test(plugin='crazy_functions.命令行助手->命令行助手', main_input='查看当前的docker容器列表')

    # plugin_test(plugin='crazy_functions.解析项目源代码->解析一个Python项目', main_input="crazy_functions/test_project/python/dqn")

    # plugin_test(plugin='crazy_functions.解析项目源代码->解析一个C项目', main_input="crazy_functions/test_project/cpp/cppipc")

    # plugin_test(plugin='crazy_functions.Latex全文润色->Latex英文润色', main_input="crazy_functions/test_project/latex/attention")

    # plugin_test(plugin='crazy_functions.批量Markdown翻译->Markdown中译英', main_input="README.md")

    # plugin_test(plugin='crazy_functions.批量翻译PDF文档_多线程->批量翻译PDF文档', main_input='crazy_functions/test_project/pdf_and_word/aaai.pdf')

    # plugin_test(plugin='crazy_functions.谷歌检索小助手->谷歌检索小助手', main_input="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=auto+reinforcement+learning&btnG=")
    
    # plugin_test(plugin='crazy_functions.总结word文档->总结word文档', main_input="crazy_functions/test_project/pdf_and_word")

    # plugin_test(plugin='crazy_functions.下载arxiv论文翻译摘要->下载arxiv论文并翻译摘要', main_input="1812.10695")

    # plugin_test(plugin='crazy_functions.联网的ChatGPT->连接网络回答问题', main_input="谁是应急食品？")

    # plugin_test(plugin='crazy_functions.解析JupyterNotebook->解析ipynb文件', main_input="crazy_functions/test_samples")

    # plugin_test(plugin='crazy_functions.数学动画生成manim->动画生成', main_input="A ball split into 2, and then split into 4, and finally split into 8.")

    # for lang in ["English", "French", "Japanese", "Korean", "Russian", "Italian", "German", "Portuguese", "Arabic"]:
    #     plugin_test(plugin='crazy_functions.批量Markdown翻译->Markdown翻译指定语言', main_input="README.md", advanced_arg={"advanced_arg": lang})

    # plugin_test(plugin='crazy_functions.Langchain知识库->知识库问答', main_input="./")

    # plugin_test(plugin='crazy_functions.Langchain知识库->读取知识库作答', main_input="What is the installation method？")

    # plugin_test(plugin='crazy_functions.Langchain知识库->读取知识库作答', main_input="远程云服务器部署？")
    
    # plugin_test(plugin='crazy_functions.Latex输出PDF结果->Latex翻译中文并重新编译PDF', main_input="2210.03629")
    
    # advanced_arg = {"advanced_arg":"--llm_to_learn=gpt-3.5-turbo --prompt_prefix='根据下面的服装类型提示，想象一个穿着者，对这个人外貌、身处的环境、内心世界、人设进行描写。要求：100字以内，用第二人称。' --system_prompt=''" }
    # plugin_test(plugin='crazy_functions.chatglm微调工具->微调数据集生成', main_input='build/dev.json', advanced_arg=advanced_arg)

    # advanced_arg = {"advanced_arg":"--pre_seq_len=128 --learning_rate=2e-2 --num_gpus=1 --json_dataset='t_code.json' --ptuning_directory='/home/hmp/ChatGLM2-6B/ptuning'     " }
    # plugin_test(plugin='crazy_functions.chatglm微调工具->启动微调', main_input='build/dev.json', advanced_arg=advanced_arg)

