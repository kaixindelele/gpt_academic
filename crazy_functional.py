from toolbox import HotReload  # HotReload 的意思是热更新，修改函数插件后，不需要重启程序，代码直接生效


def get_crazy_functions():
    from crazy_functions.读文章写摘要 import 读文章写摘要
    from crazy_functions.生成函数注释 import 批量生成函数注释
    from crazy_functions.解析项目源代码 import 解析项目本身
    from crazy_functions.解析项目源代码 import 解析一个Python项目
    from crazy_functions.解析项目源代码 import 解析一个Matlab项目
    from crazy_functions.解析项目源代码 import 解析一个C项目的头文件
    from crazy_functions.解析项目源代码 import 解析一个C项目
    from crazy_functions.解析项目源代码 import 解析一个Golang项目
    from crazy_functions.解析项目源代码 import 解析一个Rust项目
    from crazy_functions.解析项目源代码 import 解析一个Java项目
    from crazy_functions.解析项目源代码 import 解析一个前端项目
    from crazy_functions.高级功能函数模板 import 高阶功能模板函数
    from crazy_functions.Latex全文润色 import Latex英文润色
    from crazy_functions.询问多个大语言模型 import 同时问询
    from crazy_functions.解析项目源代码 import 解析一个Lua项目
    from crazy_functions.解析项目源代码 import 解析一个CSharp项目
    from crazy_functions.总结word文档 import 总结word文档
    from crazy_functions.解析JupyterNotebook import 解析ipynb文件
    from crazy_functions.对话历史存档 import 对话历史存档
    from crazy_functions.对话历史存档 import 载入对话历史存档
    from crazy_functions.对话历史存档 import 删除所有本地对话历史记录
    from crazy_functions.辅助功能 import 清除缓存
    from crazy_functions.批量Markdown翻译 import Markdown英译中
    from crazy_functions.批量总结PDF文档 import 批量总结PDF文档
    from crazy_functions.批量翻译PDF文档_多线程 import 批量翻译PDF文档
    from crazy_functions.谷歌检索小助手 import 谷歌检索小助手
    from crazy_functions.理解PDF文档内容 import 理解PDF文档内容标准文件输入
    from crazy_functions.Latex全文润色 import Latex中文润色
    from crazy_functions.Latex全文润色 import Latex英文纠错
    from crazy_functions.Latex全文翻译 import Latex中译英
    from crazy_functions.Latex全文翻译 import Latex英译中
    from crazy_functions.批量Markdown翻译 import Markdown中译英
    from crazy_functions.虚空终端 import 虚空终端


    function_plugins = {
        "虚空终端": {
            "Group": "对话|编程|学术|智能体",
            "Color": "stop",
            "AsButton": False,
            "Function": HotReload(虚空终端)
        },
        "解析整个Python项目": {
            "Group": "编程",
            "Color": "stop",
            "AsButton": False,
            "Info": "解析一个Python项目的所有源文件(.py) | 输入参数为路径",
            "Function": HotReload(解析一个Python项目)
        },
        
        "批量生成函数注释": {
            "Group": "编程",
            "Color": "stop",
            "AsButton": False,  # 加入下拉菜单中
            "Info": "批量生成函数的注释 | 输入参数为路径",
            "Function": HotReload(批量生成函数注释)
        },
        
        "本地PDF论文-精准翻译[md用vscode的Markdown PDF转PDF]": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": True,  
            "Info": "精准翻译PDF论文为中文 | 输入参数为路径",
            "Function": HotReload(批量翻译PDF文档)
        },
        
        "英文Latex项目全文润色（输入路径或上传压缩包）": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": True,  # 加入下拉菜单中
            "Info": "对英文Latex项目全文进行润色处理 | 输入参数为路径或上传压缩包",
            "Function": HotReload(Latex英文润色)
        },
        "英文Latex项目全文纠错（输入路径或上传压缩包）": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": False,  # 加入下拉菜单中
            "Info": "对英文Latex项目全文进行纠错处理 | 输入参数为路径或上传压缩包",
            "Function": HotReload(Latex英文纠错)
        },
        "中文Latex项目全文润色（输入路径或上传压缩包）": {
            "Group": "学术",
            "Color": "stop",
            "AsButton": False,  # 加入下拉菜单中
            "Info": "对中文Latex项目全文进行润色处理 | 输入参数为路径或上传压缩包",
            "Function": HotReload(Latex中文润色)
        },

        "批量Markdown中译英（输入路径或上传压缩包）": {
            "Group": "编程",
            "Color": "stop",
            "AsButton": False,  # 加入下拉菜单中
            "Info": "批量将Markdown文件中文翻译为英文 | 输入参数为路径或上传压缩包",
            "Function": HotReload(Markdown中译英)
        },
    }

    # -=--=- 尚未充分测试的实验性插件 & 需要额外依赖的插件 -=--=-
    try:
        from crazy_functions.下载arxiv论文翻译摘要 import 下载arxiv论文并翻译摘要
        function_plugins.update({
            "一键下载arxiv论文并翻译摘要（先在input输入编号，如1812.10695）": {
                "Group": "学术",
                "Color": "stop",
                "AsButton": False,  # 加入下拉菜单中
                # "Info": "下载arxiv论文并翻译摘要 | 输入参数为arxiv编号如1812.10695",
                "Function": HotReload(下载arxiv论文并翻译摘要)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.解析项目源代码 import 解析任意code项目
        function_plugins.update({
            "解析项目源代码（手动指定和筛选源代码文件类型）": {
                "Group": "编程",
                "Color": "stop",
                "AsButton": False,
                "AdvancedArgs": True,  # 调用时，唤起高级参数输入区（默认False）
                "ArgsReminder": "输入时用逗号隔开, *代表通配符, 加了^代表不匹配; 不输入代表全部匹配。例如: \"*.c, ^*.cpp, config.toml, ^*.toml\"",  # 高级参数输入区的显示提示
                "Function": HotReload(解析任意code项目)
            },
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.总结音视频 import 总结音视频
        function_plugins.update({
            "批量总结音视频（输入路径或上传压缩包）": {
                "Group": "对话",
                "Color": "stop",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "调用openai api 使用whisper-1模型, 目前支持的格式:mp4, m4a, wav, mpga, mpeg, mp3。此处可以输入解析提示，例如：解析为简体中文（默认）。",
                "Info": "批量总结音频或视频 | 输入参数为路径",
                "Function": HotReload(总结音视频)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.批量Markdown翻译 import Markdown翻译指定语言
        function_plugins.update({
            "Markdown翻译（指定翻译成何种语言）": {
                "Group": "编程",
                "Color": "stop",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder": "请输入要翻译成哪种语言，默认为Chinese。",
                "Function": HotReload(Markdown翻译指定语言)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from crazy_functions.Latex输出PDF结果 import Latex英文纠错加PDF对比
        function_plugins.update({
            "Latex英文纠错+高亮修正位置 [需Latex]": {
                "Group": "学术",
                "Color": "stop",
                "AsButton": True,
                "AdvancedArgs": True,
                "ArgsReminder": "如果有必要, 请在此处追加更细致的矫错指令（使用英文）。",
                "Function": HotReload(Latex英文纠错加PDF对比)
            }
        })
        from crazy_functions.Latex输出PDF结果 import Latex翻译中文并重新编译PDF
        function_plugins.update({
            "Arxiv论文原生翻译（输入arxivID）[需Latex]": {
                "Group": "学术",
                "Color": "stop",
                "AsButton": True,
                "AdvancedArgs": True,
                "ArgsReminder":
                    "如果有必要, 请在此处给出自定义翻译命令, 解决部分词汇翻译不准确的问题。 " +
                    "例如当单词'agent'翻译不准确时, 请尝试把以下指令复制到高级参数区: " +
                    'If the term "agent" is used in this section, it should be translated to "智能体". ',
                "Info": "Arxiv论文精细翻译 | 输入参数arxiv论文的ID，比如1812.10695",
                "Function": HotReload(Latex翻译中文并重新编译PDF)
            }
        })
        function_plugins.update({
            "本地Latex论文精细翻译（上传Latex项目）[需Latex]": {
                "Group": "学术",
                "Color": "stop",
                "AsButton": False,
                "AdvancedArgs": True,
                "ArgsReminder":
                    "如果有必要, 请在此处给出自定义翻译命令, 解决部分词汇翻译不准确的问题。 " +
                    "例如当单词'agent'翻译不准确时, 请尝试把以下指令复制到高级参数区: " +
                    'If the term "agent" is used in this section, it should be translated to "智能体". ',
                "Info": "本地Latex论文精细翻译 | 输入参数是路径",
                "Function": HotReload(Latex翻译中文并重新编译PDF)
            }
        })
    except:
        print('Load function plugin failed')

    try:
        from toolbox import get_conf
        ENABLE_AUDIO, = get_conf('ENABLE_AUDIO')
        if ENABLE_AUDIO:
            from crazy_functions.语音助手 import 语音助手
            function_plugins.update({
                "实时语音对话": {
                    "Group": "对话",
                    "Color": "stop",
                    "AsButton": True,
                    "Info": "这是一个时刻聆听着的语音对话助手 | 没有输入参数",
                    "Function": HotReload(语音助手)
                }
            })
    except:
        print('Load function plugin failed')


    """
    设置默认值:
    - 默认 Group = 对话
    - 默认 AsButton = True
    - 默认 AdvancedArgs = False
    - 默认 Color = secondary
    """
    for name, function_meta in function_plugins.items():
        if "Group" not in function_meta:
            function_plugins[name]["Group"] = '对话'
        if "AsButton" not in function_meta:
            function_plugins[name]["AsButton"] = True
        if "AdvancedArgs" not in function_meta:
            function_plugins[name]["AdvancedArgs"] = False
        if "Color" not in function_meta:
            function_plugins[name]["Color"] = 'secondary'

    return function_plugins
