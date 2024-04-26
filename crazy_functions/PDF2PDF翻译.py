from toolbox import CatchException, report_exception, get_log_folder, gen_time_str
from toolbox import update_ui, promote_file_to_downloadzone, update_ui_lastest_msg, disable_auto_promotion
from toolbox import get_conf, extract_archive
from toolbox import get_upload_folder, zip_folder
from toolbox import write_history_to_file, promote_file_to_downloadzone
from .crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from .crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
from .crazy_utils import read_and_clean_pdf_text
from .pdf_fns.parse_pdf import parse_pdf, get_avail_grobid_url, translate_pdf
from colorful import *
import copy
import os
import math
import random

@CatchException
def 批量PDF2PDF文档(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):

    disable_auto_promotion(chatbot)
    # 基本信息：功能、贡献者
    chatbot.append([
        "函数插件功能？",
        "批量翻译PDF文档。函数插件贡献者: Binary-Husky"])
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import fitz
        import tiktoken
        import scipdf
    except:
        report_execption(chatbot, history,
                         a=f"解析项目: {txt}",
                         b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade pymupdf tiktoken scipdf_parser```。")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return

    # 清空历史，以免输入溢出
    history = []

    from .crazy_utils import get_files_from_everything
    success, file_manifest, project_folder = get_files_from_everything(txt, type='.pdf')
    # 检测输入参数，如没有给定输入参数，直接退出
    if not success:
        if txt == "": txt = '空空如也的输入栏'

    # 如果没找到任何文件
    if len(file_manifest) == 0:
        report_execption(chatbot, history,
                         a=f"解析项目: {txt}", b=f"找不到任何.pdf拓展名的文件: {txt}")
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return

    # 开始正式执行任务
    DOC2X_API_KEY = get_conf("DOC2X_API_KEY")
    if len(DOC2X_API_KEY) != 0:
        yield from 解析PDF_DOC2X(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, DOC2X_API_KEY, user_request)
        return
    grobid_url = get_avail_grobid_url()
    if grobid_url is not None:
        yield from 解析PDF_基于GROBID(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, grobid_url)
    else:
        yield from update_ui_lastest_msg("GROBID服务不可用，请检查config中的GROBID_URL。作为替代，现在将执行效果稍差的旧版代码。", chatbot, history, delay=3)
        yield from 解析PDF(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)

def add_doc_black_list(result, api_key):
    print("result:", result)
    result = str(result)
    if "pages limit exceeded" in result:
        # 先读取现有的黑名单，如果没有则加进去：
        cur_black_list = []
        try:
            with open('doc_black_apis.txt', 'r') as f:
                for line in f.readlines():
                    cur_black_list.append(line.strip())
        except FileNotFoundError:
            with open('doc_black_apis.txt', 'w') as f:
                f.write("")

        if api_key not in cur_black_list:
            print("add black list:", api_key)
            with open('doc_black_apis.txt', 'a+') as f:
                f.write(api_key + '\n')

def 解析PDF_DOC2X_单文件(fp, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, DOC2X_API_KEY, user_request):

    def pdf2markdown(filepath):
        import requests, json, os
        markdown_dir = get_log_folder(plugin_name="pdf_ocr")
        doc2x_api_key = DOC2X_API_KEY

        if '\n' in doc2x_api_key[0]:
            doc2x_api_key = doc2x_api_key[0].split('\n')
            print("all_docs:", doc2x_api_key)
        if type(doc2x_api_key) is list:
            # 这里先查看是否有doc的black api
            try:
                with open('doc_black_apis.txt', 'r', encoding='utf8') as f:
                    black_apis = f.read().split('\n')
            except Exception as e:
                print("读取黑名单失败，将不会使用黑名单", e)
                black_apis = []
            avail_key_list = []
            for k in doc2x_api_key:
                # 在这儿判断这些key是否在黑名单中
                if k not in black_apis:
                    avail_key_list.append(k)
            if len(avail_key_list) > 0:
                doc2x_api_key = random.choice(avail_key_list)
                print("doc2x_api_key:", doc2x_api_key)

            else:
                raise RuntimeError("目前doc2x的api已用完，请手动去doc2x网页转编辑，然后用md翻译。")

        url = "https://api.doc2x.noedgeai.com/api/v1/pdf"

        chatbot.append((None, "加载PDF文件，发送至DOC2X解析..."))
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

        res = requests.post(
            url,
            files={"file": open(filepath, "rb")},
            data={"ocr": "1"},
            headers={"Authorization": "Bearer " + doc2x_api_key}
        )
        res_json = []
        if res.status_code == 200:
            decoded = res.content.decode("utf-8")
            for z_decoded in decoded.split('\n'):
                if len(z_decoded) == 0: continue
                assert z_decoded.startswith("data: ")
                z_decoded = z_decoded[len("data: "):]
                decoded_json = json.loads(z_decoded)
                res_json.append(decoded_json)
        else:
            raise RuntimeError(format("[ERROR] status code: %d, body: %s" % (res.status_code, res.text)))
        uuid = res_json[0]['uuid']
        if "pages limit exceeded" in res_json[0]['status']:
            add_doc_black_list(res_json[0]['status'], doc2x_api_key)
            raise RuntimeError(format("api额度用完，麻烦自己去doc2x转成md，在用MD翻译插件。[ERROR] status code: %d, body: %s, api: xxx-%s" % (res.status_code, res_json[0], doc2x_api_key[-6:])))
        to = "md" # latex, md, docx
        url = "https://api.doc2x.noedgeai.com/api/export"+"?request_id="+uuid+"&to="+to

        chatbot.append((None, f"读取解析: {url} ..."))
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

        res = requests.get(url, headers={"Authorization": "Bearer " + doc2x_api_key})
        md_zip_path = os.path.join(markdown_dir, gen_time_str() + '.zip')
        if res.status_code == 200:
            with open(md_zip_path, "wb") as f: f.write(res.content)
        else:
            raise RuntimeError(format("[ERROR] status code: %d, body: %s" % (res.status_code, res.text)))
        promote_file_to_downloadzone(md_zip_path, chatbot=chatbot)
        chatbot.append((None, f"完成解析 {md_zip_path} ..."))
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        return md_zip_path

    def deliver_to_markdown_plugin(md_zip_path, user_request):
        from crazy_functions.批量Markdown翻译 import Markdown英译中
        import shutil

        time_tag = gen_time_str()
        target_path_base = get_log_folder(chatbot.get_user())
        file_origin_name = os.path.basename(md_zip_path)
        this_file_path = os.path.join(target_path_base, file_origin_name)
        os.makedirs(target_path_base, exist_ok=True)
        shutil.copyfile(md_zip_path, this_file_path)
        ex_folder = this_file_path + ".extract"
        extract_archive(
            file_path=this_file_path, dest_dir=ex_folder
        )
        chatbot.append((None, f"调用Markdown插件 {ex_folder} ..."))
        plugin_kwargs['markdown_expected_output_dir'] = ex_folder

        translated_f_name = 'translated_markdown.md'
        generated_fp = plugin_kwargs['markdown_expected_output_path'] = os.path.join(ex_folder, translated_f_name)
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        yield from Markdown英译中(ex_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request)
        if os.path.exists(generated_fp):
            # 修正一些公式问题
            with open(generated_fp, 'r', encoding='utf8') as f:
                content = f.read()
            # 将公式中的\[ \]替换成$$
            content = content.replace(r'\[', r'$$').replace(r'\]', r'$$')
            # 将公式中的\( \)替换成$
            content = content.replace(r'\(', r'$').replace(r'\)', r'$')
            content = content.replace('```', '\n').replace('```markdown', '\n')
            with open(generated_fp, 'w', encoding='utf8') as f:
                f.write(content)
            # 生成包含图片的压缩包
            dest_folder = get_log_folder(chatbot.get_user())
            zip_name = '翻译后的带图文档-只用下载它即可.zip'
            zip_folder(source_folder=ex_folder, dest_folder=dest_folder, zip_name=zip_name)
            zip_fp = os.path.join(dest_folder, zip_name)
            # 生成在线预览html
            file_name = '在线预览翻译' + gen_time_str() + '.html'
            with open('crazy_functions/pdf_fns/report_template_v2.html', 'r', encoding='utf8') as f:
                html_template = f.read()
            html_template = html_template.replace("{MARKDOWN_FILE_PATH}", translated_f_name)
            preview_fp = os.path.join(ex_folder, file_name)
            with open(preview_fp, 'w', encoding='utf8') as f:
                f.write(html_template)
            # promote_file_to_downloadzone(preview_fp, chatbot=chatbot)
            promote_file_to_downloadzone(zip_fp, chatbot=chatbot)
            yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
    md_zip_path = yield from pdf2markdown(fp)
    yield from deliver_to_markdown_plugin(md_zip_path, user_request)

def 解析PDF_DOC2X(file_manifest, *args):
    for index, fp in enumerate(file_manifest):
        yield from 解析PDF_DOC2X_单文件(fp, *args)
    return



def 解析PDF_基于GROBID(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, grobid_url):
    import copy, json
    TOKEN_LIMIT_PER_FRAGMENT = 1024
    generated_conclusion_files = []
    generated_html_files = []
    DST_LANG = "中文"
    from crazy_functions.crazy_utils import construct_html
    for index, fp in enumerate(file_manifest):
        chatbot.append(["当前进度：", f"正在连接GROBID服务，请稍候: {grobid_url}\n如果等待时间过长，请修改config中的GROBID_URL，可修改成本地GROBID服务。"]); yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
        article_dict = parse_pdf(fp, grobid_url)
        grobid_json_res = os.path.join(get_log_folder(), gen_time_str() + "grobid.json")
        with open(grobid_json_res, 'w+', encoding='utf8') as f:
            f.write(json.dumps(article_dict, indent=4, ensure_ascii=False))
        promote_file_to_downloadzone(grobid_json_res, chatbot=chatbot)

        if article_dict is None: raise RuntimeError("解析PDF失败，请检查PDF是否损坏。")
        yield from translate_pdf(article_dict, llm_kwargs, chatbot, fp, generated_conclusion_files, TOKEN_LIMIT_PER_FRAGMENT, DST_LANG)
    chatbot.append(("给出输出文件清单", str(generated_conclusion_files + generated_html_files)))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面


def 解析PDF(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    """
    此函数已经弃用
    """
    import copy
    TOKEN_LIMIT_PER_FRAGMENT = 256
    generated_conclusion_files = []
    generated_html_files = []
    from crazy_functions.crazy_utils import construct_html
    for index, fp in enumerate(file_manifest):
        # 读取PDF文件
        file_content, page_one = read_and_clean_pdf_text(fp)
        file_content = file_content.encode('utf-8', 'ignore').decode()   # avoid reading non-utf8 chars
        page_one = str(page_one).encode('utf-8', 'ignore').decode()      # avoid reading non-utf8 chars

        # 递归地切割PDF文件
        from .crazy_utils import breakdown_txt_to_satisfy_token_limit_for_pdf
        from request_llm.bridge_all import model_info
        enc = model_info["gpt-3.5-turbo"]['tokenizer']
        def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))
        paper_fragments = breakdown_txt_to_satisfy_token_limit_for_pdf(
            txt=file_content,  get_token_fn=get_token_num, limit=TOKEN_LIMIT_PER_FRAGMENT)
        page_one_fragments = breakdown_txt_to_satisfy_token_limit_for_pdf(
            txt=page_one, get_token_fn=get_token_num, limit=TOKEN_LIMIT_PER_FRAGMENT//4)

        # 为了更好的效果，我们剥离Introduction之后的部分（如果有）
        paper_meta = page_one_fragments[0].split('introduction')[0].split('Introduction')[0].split('INTRODUCTION')[0]

        # 单线，获取文章meta信息
        paper_meta_info = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=f"以下是一篇学术论文的基础信息，请从中提取出“标题”、“收录会议或期刊”、“作者”、“摘要”、“编号”、“作者邮箱”这六个部分。请用markdown格式输出，最后用中文翻译摘要部分，不要提取Introduction部分的内容。请提取：{paper_meta}",
            inputs_show_user=f"请从{fp}中提取出“标题”、“收录会议或期刊”等基本信息。",
            llm_kwargs=llm_kwargs,
            chatbot=chatbot, history=[],
            sys_prompt="Your job is to collect information from materials。",
        )

        # 多线，翻译
        gpt_response_collection = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array=[
                f"你需要翻译以下内容：\n{frag}" for frag in paper_fragments],
            inputs_show_user_array=[f"\n---\n 原文： \n\n {frag.replace('#', '')}  \n---\n 翻译：\n " for frag in paper_fragments],
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            history_array=[[paper_meta] for _ in paper_fragments],
            sys_prompt_array=[
                "请你作为一个学术翻译，负责把学术论文的部分章节文本，准确翻译成中文，并且按照规范的中文表述。注意：1. 文章中的每一句话都要翻译，并且消除输入文本前后的无意义乱码，2. 请自动识别小章节标题(小标题长度不要超过20个字符，也不要少于3个字符)，并且用'### xxx'的markdown格式标记出来。" for _ in paper_fragments],
            # max_workers=5  # OpenAI所允许的最大并行过载
        )
        gpt_response_collection_md = copy.deepcopy(gpt_response_collection)
        # 整理报告的格式
        add_origin = True
        for i, k in enumerate(gpt_response_collection_md):
            if i % 2 ==0:
                cur_trans = gpt_response_collection_md[i]
                # 做个小小的处理，把翻译的结果中非常长的“#”去掉
                temp_trans = ""
                for line_text in cur_trans.split('\n'):
                    if len(line_text) == 0:
                        # print("空行")
                        temp_trans += "\n\n"
                    else:
                        if "#" in line_text[0]:
                            if len(line_text.split(' ')) > 12:
                                temp_trans += line_text.replace('#', '')
                            else:
                                temp_trans += line_text
                                temp_trans += "\n\n"
                        else:
                            temp_trans += line_text + "\n\n"

                # gpt_response_collection_md[i] = f"\n\n---\n\n ## 原文[{i//2}/{len(gpt_response_collection_md)//2}]： \n\n {paper_fragments[i//2].replace('#', '')}  \n\n---\n\n ## 翻译[{i//2}/{len(gpt_response_collection_md)//2}]：\n "
                if add_origin:
                    gpt_response_collection_md[i] = f"\n\n---\n\n ## 原文[{i//2}/{len(gpt_response_collection_md)//2}]： \n\n {paper_fragments[i//2].replace('#', '')}  \n\n---\n\n ## 翻译[{i//2}/{len(gpt_response_collection_md)//2}]：\n "
                else:
                    gpt_response_collection_md[i] = ""
            else:
                cur_trans = gpt_response_collection_md[i]
                # 做个小小的处理，把翻译的结果中非常长的“#”去掉
                temp_trans = ""
                for line_text in cur_trans.split('\n'):
                    if len(line_text) == 0:
                        # print("空行")
                        temp_trans += "\n\n"
                    else:
                        if "#" in line_text[0]:
                            if len(line_text) > 12:
                                temp_trans += line_text.replace('#', '')
                            else:
                                temp_trans += line_text
                                temp_trans += "\n\n"
                        else:
                            temp_trans += line_text + "\n\n"

                gpt_response_collection_md[i] = temp_trans
        final = ["一、论文概况\n\n---\n\n", paper_meta_info.replace('# ', '### ') + '\n\n---\n\n', "二、论文翻译", ""]
        final.extend(gpt_response_collection_md)
        create_report_file_name = f"{os.path.basename(fp)}.trans.md"
        res = write_history_to_file(final, create_report_file_name)
        promote_file_to_downloadzone(res, chatbot=chatbot)

        # 更新UI
        generated_conclusion_files.append(f'{get_log_folder()}/{create_report_file_name}')
        chatbot.append((f"{fp}完成了吗？", res))
        yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

        # write html
        try:
            ch = construct_html()
            orig = ""
            trans = ""
            gpt_response_collection_html = copy.deepcopy(gpt_response_collection)
            for i,k in enumerate(gpt_response_collection_html):
                if i%2==0:
                    gpt_response_collection_html[i] = paper_fragments[i//2].replace('#', '')
                else:
                    gpt_response_collection_html[i] = gpt_response_collection_html[i]
            final = ["论文概况", paper_meta_info.replace('# ', '### '),  "二、论文翻译",  ""]
            final.extend(gpt_response_collection_html)
            for i, k in enumerate(final):
                if i%2==0:
                    orig = k
                if i%2==1:
                    trans = k
                    ch.add_row(a=orig, b=trans)
            create_report_file_name = f"{os.path.basename(fp)}.trans.html"
            generated_html_files.append(ch.save_file(create_report_file_name))
        except:
            from toolbox import trimmed_format_exc
            print('writing html result failed:', trimmed_format_exc())

    # 准备文件的下载
    for pdf_path in generated_conclusion_files:
        # 重命名文件
        rename_file = f'翻译-{os.path.basename(pdf_path)}'
        promote_file_to_downloadzone(pdf_path, rename_file=rename_file, chatbot=chatbot)
    for html_path in generated_html_files:
        # 重命名文件
        rename_file = f'翻译-{os.path.basename(html_path)}'
        promote_file_to_downloadzone(html_path, rename_file=rename_file, chatbot=chatbot)
    chatbot.append(("给出输出文件清单", str(generated_conclusion_files + generated_html_files)))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面


