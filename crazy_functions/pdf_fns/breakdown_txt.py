import sys
import re
sys.path.append('/gpt/gpt_academic')
from crazy_functions.ipc_fns.mp import run_in_subprocess_with_timeout

def replace_periods(txt):
    return re.sub(r'(?<!\d)\.(?!\d)', '。\n', txt)

def find_next_section(text):    
    pattern = r'\n[A-Z]'
    match = re.search(pattern, text)
    if match:
        return match.start()
    else:
        return 0
    
def replace_newlines(text):
    punctuation = ".?!"
    uppercase_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = []
    i = 0
    
    while i < len(text) - 1:
        if text[i] == '\n':
            if text[i - 1] not in punctuation and text[i + 1] not in uppercase_letters:
                result.append(' ')
            else:
                result.append(text[i])
        else:
            result.append(text[i])
        i += 1
    result.append(text[-1])
    return ''.join(result)


def is_chinese(text):
    chinese_regex = "[\u4e00-\u9fa5]"
    total_chars = 0
    chinese_chars = 0

    # 取前30个字符
    start_text = text[:30]
    total_chars += len(start_text)
    chinese_chars += len(re.findall(chinese_regex, start_text))

    # 取后30个字符
    end_text = text[-30:]
    total_chars += len(end_text)
    chinese_chars += len(re.findall(chinese_regex, end_text))

    # 计算中文字符比例
    chinese_ratio = chinese_chars / total_chars

    return chinese_ratio > 0.7

def force_breakdown(txt, limit, get_token_fn):
    """ 
    当无法用标点、空行分割时，使用最暴力的方法切割文本
    从后往前遍历，找到第一个满足token数限制的切分点
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktoken未知错误", "Tiktoken未知错误"


def maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage):
    """ 
    为了加速计算，采用特殊手段管理文本存储
    当待切分文本过长时，将超出部分转存到storage中
    当待切分文本过短时，从storage中取回部分文本
    """
    _min = int(5e4)  # 最小阈值
    _max = int(1e5)  # 最大阈值
    
    # 如果待切分文本小于最小阈值且storage不为空，从storage取回文本
    if len(remain_txt_to_cut) < _min and len(remain_txt_to_cut_storage) > 0:
        remain_txt_to_cut = remain_txt_to_cut + remain_txt_to_cut_storage
        remain_txt_to_cut_storage = ""
    
    # 如果待切分文本大于最大阈值，将超出部分转存到storage
    if len(remain_txt_to_cut) > _max:
        remain_txt_to_cut_storage = remain_txt_to_cut[_max:] + remain_txt_to_cut_storage
        remain_txt_to_cut = remain_txt_to_cut[:_max]
    
    return remain_txt_to_cut, remain_txt_to_cut_storage


def doc2json(limit, get_token_fn, text, steps=[400, 200, 100]):
    """ 
    直接重新定义一个doc2x的切分函数。
    切分原则：
    1. 优先从## xxx到下一个## 之间的内容。
    2. 如果当前文本中有<table>，那么要确保<\table>在本文段中。
    """
    # 这里还需要确保，里面没有超过##的章节！
    print("进入doc2x的文本切分模式：")
    main_text_tokens = get_token_fn(text)
    print("total tokens:", main_text_tokens)
    doc_list = text.split("## ")
    # 然后再找标题：单个"# ""
    doc_json = {}
    if "# " in doc_list[0]:
        # 这时候，先把标题拿到，规则是带#的那一行就是：
        # 先用# 来切分完整的字符串：
        title = doc_list[0].strip().split("# ")[-1].split("\n")[0]
        # # 这时候会拿到前后两个信息
        # title = doc_list[0].strip().split("\n")[0].split('# ')[1]
        print("title:", title)
        doc_json.update({"title": title})
        pre_content = doc_list[0].strip().split("# "+title)[1]
        print("pre_content:", pre_content)
        doc_json.update({"pre_content": pre_content})
    # 接下来就是任意的正文章节：对于每个章节，都先提取对应的标题，然后再提取内容。
    for sec in doc_list[1:]:
        section_name = sec.strip().split("\n")[0]
        print("section_name:", section_name)
        section_content = sec.strip().split(section_name)[1]
        # 存在一些特殊的总章节，没有正文。
        print("section_content:", section_content[:100], "...")
        doc_json.update({section_name: section_content})    
    return doc_json


def doc_sec_cut(limit, get_token_fn, sec_content):
    # 先对章节进行切分，分成多个列表：
    # 先按两个换行切分，看是否满足要求。
    sec_list = sec_content.strip().split("\n\n")
    if len([True for sec in sec_list if get_token_fn(sec) < limit]) == len(sec_list):
        print("直接按照两个换行切分，已完成！")
        return sec_list
    else:
        # 再按照一个换行切分。
        all_sec_list = []
        for sec in sec_list:
            if get_token_fn(sec) < limit:
                if sec.strip():
                    all_sec_list.append(sec)
            else:
                new_sec_list = sec.split("\n")
                for new_sec in new_sec_list:
                    if new_sec.strip():
                        all_sec_list.append(new_sec)
        # 到这一步，可以获得所有的段落列表，接下来就不断叠加。
        # 叠加规则是，没凑够最大值，就加一个元素。
        final_list = []
        cur_text = ""
        print("总共的段落数：", len(all_sec_list))
        for sec in all_sec_list:
            if get_token_fn(cur_text) + get_token_fn(sec) < limit:
                cur_text += sec
            else:
                final_list.append(cur_text)
                cur_text = sec
        print("最终的段落数：", len(final_list))
        return final_list

def doc2x_cut(limit, get_token_fn, doc_json, steps=[400, 200, 100]):
    """ 
    直接重新定义一个doc2x的切分函数。
    切分原则：
    1. 优先从## xxx到下一个## 之间的内容。
    2. 如果当前文本中有<table>，那么要确保<\table>在本文段中。
    """
    print("进入doc2x的文本切分模式：")
    text = str(doc_json)
    main_text_tokens = get_token_fn(text)
    print("total tokens:", main_text_tokens)
    # 按照段落来切分，然后再判断每个段落是否超过限制，如果超过限制，再简单切分一下。
    final_list = []
    last_sec_name = ""
    for sec_name, sec_content in doc_json.items():
        if sec_name == "title":
            final_list.append("# " + sec_content)
        elif sec_name == "pre_content":
            final_list.append(sec_content)            
        # 然后再切分其他的正文章节
        else:
            if get_token_fn(sec_content) < limit:
                # 如果当前的section没有内容，则把标题放到下一个section中。
                if sec_content.strip() == "":
                    last_sec_name = sec_name
                    continue
                else:
                    last_sec_name = ''                        
                # print("current_section:", sec_content[:100], "...")
                if last_sec_name != "":
                    final_list.append("## \n\n"+last_sec_name+"\n\n## "+sec_name+"\n\n"+sec_content)
                else:
                    final_list.append("## "+sec_name+"\n\n"+sec_content)
            else:
                # 开始切分：切分规则是根据\n来。
                new_list = doc_sec_cut(limit, get_token_fn, sec_content)
                
                for index, new_sec in enumerate(new_list):
                    if index == 0:
                        final_list.append("## "+sec_name+"\n\n"+new_sec+"\n")
                    else:
                        final_list.append(new_sec+"\n")
            
    return final_list


def cut(limit, get_token_fn, txt_tocut, must_break_at_empty_line, break_anyway=False):
    """ 
    文本切分的主要函数
    """
    res = []  # 存储切分后的文本片段
    total_len = len(txt_tocut)
    fin_len = 0  # 已处理的文本长度
    remain_txt_to_cut = txt_tocut
    remain_txt_to_cut_storage = ""

    # 初始化存储
    remain_txt_to_cut, remain_txt_to_cut_storage = maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage)

    while True:
        # 如果剩余文本的token数小于等于限制，直接添加到结果中并结束循环
        if get_token_fn(remain_txt_to_cut) <= limit:
            res.append(remain_txt_to_cut)
            fin_len += len(remain_txt_to_cut)
            break
        else:
            # 如果剩余文本的token数大于限制，需要进行切分
            lines = remain_txt_to_cut.split('\n')

            # 估计一个初始切分点
            estimated_line_cut = int(limit / get_token_fn(remain_txt_to_cut) * len(lines))

            # 从估计的切分点开始，向前查找合适的切分点
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    # 如果要求在空行处切分，跳过非空行
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break

            # 如果没有找到合适的切分点
            if cnt == 0:
                if break_anyway:
                    # 如果允许暴力切分，使用force_breakdown函数
                    print("开启暴力切分！")
                    prev, post = force_breakdown(remain_txt_to_cut, limit, get_token_fn)
                else:
                    # 不允许暴力切分，抛出异常
                    raise RuntimeError(f"存在一行极长的文本！{remain_txt_to_cut}")

            # 将切分的前半部分添加到结果中
            res.append(prev)
            fin_len += len(prev)
            
            # 准备下一次迭代
            remain_txt_to_cut = post
            remain_txt_to_cut, remain_txt_to_cut_storage = maintain_storage(remain_txt_to_cut, remain_txt_to_cut_storage)
            
            # 打印处理进度
            process = fin_len/total_len
            print(f'正在文本切分 {int(process*100)}%')
            
            # 如果剩余文本为空，结束循环
            if len(remain_txt_to_cut.strip()) == 0:
                break
    return res


def breakdown_text_to_satisfy_token_limit_(txt, limit, llm_model="gpt-3.5-turbo"):
    """ 
    使用多种方式尝试切分文本，以满足token限制
    按照优先级依次尝试不同的切分方法
    """
    from request_llms.bridge_all import model_info
    enc = model_info[llm_model]['tokenizer']
    def get_token_fn(txt): return len(enc.encode(txt, disallowed_special=()))

    # 如果是doc2x的Markdown格式，则直接进入doc2x的切分函数：
    if '</table>' in txt or "](images" in txt:
        pdf_json = doc2json(limit, get_token_fn, txt)
        final_list = doc2x_cut(limit, get_token_fn, pdf_json)
        return final_list
    else:
        try:
            # 第1次尝试：将双空行（\n\n）作为切分点
            return cut(limit, get_token_fn, txt, must_break_at_empty_line=True)
        except RuntimeError:
            try:
                # 第2次尝试：将单空行（\n）作为切分点
                return cut(limit, get_token_fn, txt, must_break_at_empty_line=False)
            except RuntimeError:
                try:
                    # 第3次尝试：将英文句号（.）作为切分点            
                    res = cut(limit, get_token_fn, 
                            replace_periods(txt), 
                            must_break_at_empty_line=False)
                    last_res = [r.replace('。\n', '.') for r in res]
                    return last_res
                except RuntimeError:
                    try:
                        # 第4次尝试：将中文句号（。）作为切分点
                        res = cut(limit, get_token_fn, txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                        return [r.replace('。。\n', '。') for r in res]
                    except RuntimeError:
                        # 第5次尝试：无法找到合适的切分点，强制切分
                        return cut(limit, get_token_fn, txt, must_break_at_empty_line=False, break_anyway=True)

# 使用子进程运行breakdown_text_to_satisfy_token_limit_函数，设置超时时间为60秒
breakdown_text_to_satisfy_token_limit = run_in_subprocess_with_timeout(breakdown_text_to_satisfy_token_limit_, timeout=60)

# 主函数，用于测试
if __name__ == '__main__':
    
    from crazy_functions.crazy_utils import read_and_clean_pdf_text
    file_content, page_one = read_and_clean_pdf_text("demo.pdf")
    
    # 直接测试doc2x的Markdown文本：
    with open('demo.md', 'r', encoding='utf-8', errors='replace') as f:
        file_content = f.read()

    print(len(file_content))
    TOKEN_LIMIT_PER_FRAGMENT = 1024
    from request_llms.bridge_all import model_info
    llm_model="gpt-3.5-turbo"
    enc = model_info[llm_model]['tokenizer']
    def get_token_fn(txt): return len(enc.encode(txt, disallowed_special=()))

    pdf_json = doc2json(TOKEN_LIMIT_PER_FRAGMENT, get_token_fn, file_content)
    final_list = doc2x_cut(TOKEN_LIMIT_PER_FRAGMENT, get_token_fn, pdf_json)
    for sec in final_list:
        print("sec:", sec)
        print("=====================================")
    # for key, value in pdf_json.items():
    #     print(key, value)
    # res = breakdown_text_to_satisfy_token_limit_(file_content, TOKEN_LIMIT_PER_FRAGMENT)
    # for r in res:    
    #     print("res:\n", r)
    #     print("=========================")