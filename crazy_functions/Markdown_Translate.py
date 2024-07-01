import glob, shutil, os, re, logging
import json
from toolbox import update_ui, trimmed_format_exc, gen_time_str, disable_auto_promotion
from toolbox import CatchException, report_exception, get_log_folder
from toolbox import write_history_to_file, promote_file_to_downloadzone
fast_debug = False

class PaperFileGroup():
    def __init__(self):
        self.file_paths = []
        self.file_contents = []
        self.sp_file_contents = []
        self.sp_file_index = []
        self.sp_file_tag = []

        # count_token
        from request_llms.bridge_all import model_info
        enc = model_info["gpt-3.5-turbo"]['tokenizer']
        def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))
        self.get_token_num = get_token_num

    def run_file_split(self, max_token_limit=2048):
        """
        将长文本分离开来
        """
        for index, file_content in enumerate(self.file_contents):
            if self.get_token_num(file_content) < max_token_limit:
                self.sp_file_contents.append(file_content)
                self.sp_file_index.append(index)
                self.sp_file_tag.append(self.file_paths[index])
            else:
                from crazy_functions.pdf_fns.breakdown_txt import breakdown_text_to_satisfy_token_limit
                segments = breakdown_text_to_satisfy_token_limit(file_content, max_token_limit)
                for j, segment in enumerate(segments):
                    self.sp_file_contents.append(segment)
                    self.sp_file_index.append(index)
                    self.sp_file_tag.append(self.file_paths[index] + f".part-{j}.md")
        logging.info('Segmentation: done')

    def merge_result(self):
        self.file_result = ["" for _ in range(len(self.file_paths))]
        for r, k in zip(self.sp_file_result, self.sp_file_index):
            self.file_result[k] += r

    def write_result(self, language):
        manifest = []
        for path, res in zip(self.file_paths, self.file_result):
            dst_file = os.path.join(get_log_folder(), f'{gen_time_str()}.md')
            with open(dst_file, 'w', encoding='utf8') as f:
                manifest.append(dst_file)
                f.write(res)
        return manifest

def extract_dict_from_string(term_str):
    dict_pattern = re.compile(r'{.*}', re.DOTALL)
    dict_match = dict_pattern.search(term_str)

    if dict_match:
        dict_str = dict_match.group().replace('\n', '')
        term_dict = eval(dict_str)
        return term_dict
    else:
        return {}

def extract_exclude_dict_from_string(term_str):
    dict_pattern = re.compile(r'{.*}', re.DOTALL)
    dict_match = dict_pattern.search(term_str)

    if dict_match:
        dict_str = dict_match.group().replace('\n', '')
        return term_str.replace(dict_str, '')
    else:
        return term_str

def 多文件翻译(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, language='en'):
    from .crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency

    #  <-------- 读取Markdown文件，删除其中的所有注释 ---------->
    pfg = PaperFileGroup()

    for index, fp in enumerate(file_manifest):
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()
            # 记录删除注释后的文本
            pfg.file_paths.append(fp)
            pfg.file_contents.append(file_content)

    #  <-------- 拆分过长的Markdown文件 ---------->
    pfg.run_file_split(max_token_limit=2048)
    n_split = len(pfg.sp_file_contents)

    more_req = plugin_kwargs.get("advanced_arg", "")
    # 提取术语的字典和剩余指令
    user_term_dict = extract_dict_from_string(more_req)
    user_prompt = extract_exclude_dict_from_string(more_req)
    # 如果有术语，但没有提示词，则默认给一个提示词：
    if len(user_term_dict) > 0:
        if len(user_prompt.strip())==0:
            user_prompt = "基于上面的术语库，把对应的论文章节翻译成地道的中文表达，并且保持格式的准确性"

    # 读取本地默认术语
    with open('all_terms.json', 'r') as file:
        default_term_dict = json.load(file)

    # 访问数据
    print("default_term_dict:", default_term_dict)

    # 合并两个术语字典：
    default_term_dict.update(user_term_dict)

    print("more_req:", more_req)
    if len(more_req) == "":
        more_req = ''
    else:
        if '```' not in more_req:
            more_req = f"```{more_req}```"

    #  <-------- 多线程翻译开始 ---------->
    if language == 'en->zh':
        if len(more_req) == 0:
            inputs_array = ["This is a Markdown file, translate it into Chinese, do NOT modify any existing Markdown commands" +
                            f"\n\n{frag}" for frag in pfg.sp_file_contents]
            inputs_show_user_array = [f"翻译 {f}" for f in pfg.sp_file_tag]
            sys_prompt_array = ["You are a professional academic paper translator."  + plugin_kwargs.get("additional_prompt", "") for _ in range(n_split) for _ in range(n_split)]
        else:
            # inputs_array = ["This is a Markdown file, translate it into Chinese, do NOT modify any existing Markdown commands, do NOT use code wrapper (```), and you should follow this requirement:" + str(more_req) + "\n The text is " +
            #                 f"\n\n{frag}" for frag in pfg.sp_file_contents]
            inputs_array = [f"""
            This is a Markdown paper paragraph text, you should translate it into authentic Chinese based on the following terms: {more_req}.\n
                            Some requests for translation are as follows:
                            - Please keep these terms accurate when translating.
                            - Please keep the chapter title text accurate and clear.
                            - Please keep the accuracy of the output format in Markdown format.
                            \n
                            Your output format is
                            ```markdown
                            translated text.
                            ```
                            The actual Markdown paper paragraph text you want to translate is as follows: ```{frag}```.\n
                            """ for frag in pfg.sp_file_contents]

            # 这里的列表就得详细的循环：
            inputs_array = []
            for frag in pfg.sp_file_contents:
                cur_term = {}
                for key, value in default_term_dict.items():
                    if key.lower() in frag.lower():
                        cur_term.update({key:value})
                print("cur_term:", cur_term)
                cur_term = '```' + str(cur_term) + '```'
                cur_input = f"""
                This is a Markdown paper paragraph text, you should translate it into authentic Chinese based on the following terms: {cur_term}\n

                {user_prompt}.\n
                                Some requests for translation are as follows:
                                - Please keep these terms accurate when translating.
                                - Please keep the chapter title text accurate and clear.
                                - Please keep the accuracy of the output format in Markdown format.
                                \n
                                Your output format is
                                ```markdown
                                translated text.
                                ```
                                The actual Markdown paper paragraph text you want to translate is as follows: ```{frag}```.\n
                                """
                inputs_array.append(cur_input)

            inputs_show_user_array = [f"翻译 {f}" for f in pfg.sp_file_tag]
            sys_prompt_array = ["You are a professional academic paper translator."  + plugin_kwargs.get("additional_prompt", "") for _ in range(n_split) for _ in range(n_split)]
    elif language == 'zh->en':
        inputs_array = [f"This is a Markdown file, translate it into English, do NOT modify any existing Markdown commands, do NOT use code wrapper (```), ONLY answer me with translated results:" +
                        f"\n\n{frag}" for frag in pfg.sp_file_contents]
        inputs_show_user_array = [f"翻译 {f}" for f in pfg.sp_file_tag]
        sys_prompt_array = ["You are a professional academic paper translator." + plugin_kwargs.get("additional_prompt", "") for _ in range(n_split)]
    else:
        inputs_array = [f"This is a Markdown file, translate it into {language}, do NOT modify any existing Markdown commands, do NOT use code wrapper (```), ONLY answer me with translated results:" +
                        f"\n\n{frag}" for frag in pfg.sp_file_contents]
        inputs_show_user_array = [f"翻译 {f}" for f in pfg.sp_file_tag]
        sys_prompt_array = ["You are a professional academic paper translator." + plugin_kwargs.get("additional_prompt", "") for _ in range(n_split)]

    gpt_response_collection = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
        inputs_array=inputs_array,
        inputs_show_user_array=inputs_show_user_array,
        llm_kwargs=llm_kwargs,
        chatbot=chatbot,
        history_array=[[""] for _ in range(n_split)],
        sys_prompt_array=sys_prompt_array,
        # max_workers=5,  # OpenAI所允许的最大并行过载
        scroller_max_len = 80
    )
    try:
        pfg.sp_file_result = []
        for i_say, gpt_say in zip(gpt_response_collection[0::2], gpt_response_collection[1::2]):
            # gpt_say = gpt_say.strip().replace("```markdown", "").replace("```", "")
            pfg.sp_file_result.append(gpt_say)
        pfg.merge_result()
        # pfg.write_result(language)
        output_file_arr = pfg.write_result(language)
        for output_file in output_file_arr:
            promote_file_to_downloadzone(output_file, chatbot=chatbot)
            if 'markdown_expected_output_path' in plugin_kwargs:
                expected_f_name = plugin_kwargs['markdown_expected_output_path']
                shutil.copyfile(output_file, expected_f_name)
    except:
        logging.error(trimmed_format_exc())

    #  <-------- 整理结果，退出 ---------->
    create_report_file_name = gen_time_str() + f"-chatgpt.md"
    res = write_history_to_file(gpt_response_collection, file_basename=create_report_file_name)
    promote_file_to_downloadzone(res, chatbot=chatbot)
    history = gpt_response_collection
    chatbot.append((f"{fp}完成了吗？", res))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面


def get_files_from_everything(txt, preference=''):
    if txt == "": return False, None, None
    success = True
    if txt.startswith('http'):
        import requests
        from toolbox import get_conf
        proxies = get_conf('proxies')
        # 网络的远程文件
        if preference == 'Github':
            logging.info('正在从github下载资源 ...')
            if not txt.endswith('.md'):
                # Make a request to the GitHub API to retrieve the repository information
                url = txt.replace("https://github.com/", "https://api.github.com/repos/") + '/readme'
                response = requests.get(url, proxies=proxies)
                txt = response.json()['download_url']
            else:
                txt = txt.replace("https://github.com/", "https://raw.githubusercontent.com/")
                txt = txt.replace("/blob/", "/")

        r = requests.get(txt, proxies=proxies)
        download_local = f'{get_log_folder(plugin_name="批量Markdown翻译")}/raw-readme-{gen_time_str()}.md'
        project_folder = f'{get_log_folder(plugin_name="批量Markdown翻译")}'
        with open(download_local, 'wb+') as f: f.write(r.content)
        file_manifest = [download_local]
    elif txt.endswith('.md'):
        # 直接给定文件
        file_manifest = [txt]
        project_folder = os.path.dirname(txt)
    elif os.path.exists(txt):
        # 本地路径，递归搜索
        project_folder = txt
        file_manifest = [f for f in glob.glob(f'{project_folder}/**/*.md', recursive=True)]
    else:
        project_folder = None
        file_manifest = []
        success = False

    return success, file_manifest, project_folder


@CatchException
def Markdown英译中(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    # 基本信息：功能、贡献者
    chatbot.append([
        "函数插件功能？",
        "对整个Markdown项目进行翻译。函数插件贡献者: Binary-Husky"])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import tiktoken
    except:
        report_exception(chatbot, history,
                         a=f"解析项目: {txt}",
                         b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade tiktoken```。")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    history = []    # 清空历史，以免输入溢出

    success, file_manifest, project_folder = get_files_from_everything(txt, preference="Github")

    if not success:
        # 什么都没有
        if txt == "": txt = '空空如也的输入栏'
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到本地项目或无权访问: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return

    if len(file_manifest) == 0:
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到任何.md文件: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return

    yield from 多文件翻译(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, language='en->zh')


@CatchException
def Markdown中译英(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    # 基本信息：功能、贡献者
    chatbot.append([
        "函数插件功能？",
        "对整个Markdown项目进行翻译。函数插件贡献者: Binary-Husky"])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import tiktoken
    except:
        report_exception(chatbot, history,
                         a=f"解析项目: {txt}",
                         b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade tiktoken```。")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    history = []    # 清空历史，以免输入溢出
    success, file_manifest, project_folder = get_files_from_everything(txt)
    if not success:
        # 什么都没有
        if txt == "": txt = '空空如也的输入栏'
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到本地项目或无权访问: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    if len(file_manifest) == 0:
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到任何.md文件: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    yield from 多文件翻译(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, language='zh->en')


@CatchException
def Markdown翻译指定语言(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    # 基本信息：功能、贡献者
    chatbot.append([
        "函数插件功能？",
        "对整个Markdown项目进行翻译。函数插件贡献者: Binary-Husky"])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import tiktoken
    except:
        report_exception(chatbot, history,
                         a=f"解析项目: {txt}",
                         b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade tiktoken```。")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    history = []    # 清空历史，以免输入溢出
    success, file_manifest, project_folder = get_files_from_everything(txt)
    if not success:
        # 什么都没有
        if txt == "": txt = '空空如也的输入栏'
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到本地项目或无权访问: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return
    if len(file_manifest) == 0:
        report_exception(chatbot, history, a = f"解析项目: {txt}", b = f"找不到任何.md文件: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return

    if ("advanced_arg" in plugin_kwargs) and (plugin_kwargs["advanced_arg"] == ""): plugin_kwargs.pop("advanced_arg")
    language = plugin_kwargs.get("advanced_arg", 'Chinese')
    yield from 多文件翻译(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, language=language)