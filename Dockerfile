# 此Dockerfile适用于“无本地模型”的环境构建，如果需要使用chatglm等本地模型，请参考 docs/Dockerfile+ChatGLM
# - 1 修改 `config.py`
# - 2 构建 docker build -t gpt-academic-nolocal-latex -f docs/GithubAction+NoLocal+Latex .
# - 3 运行 docker run -v /home/fuqingxu/arxiv_cache:/root/arxiv_cache --rm -it --net=host gpt-academic-nolocal-latex

FROM ghcr.io/binary-husky/gpt_academic_with_all_capacity:master
ENV PATH "$PATH:/usr/local/texlive/2022/bin/x86_64-linux"
ENV PATH "$PATH:/usr/local/texlive/2023/bin/x86_64-linux"
ENV PATH "$PATH:/usr/local/texlive/2024/bin/x86_64-linux"
ENV PATH "$PATH:/usr/local/texlive/2025/bin/x86_64-linux"
ENV PATH "$PATH:/usr/local/texlive/2026/bin/x86_64-linux"

# 指定路径
WORKDIR /gpt

RUN pip3 install openai numpy arxiv rich
RUN pip3 install colorama Markdown pygments pymupdf
RUN pip3 install python-docx pdfminer 
RUN pip3 install nougat-ocr

# 装载项目文件
COPY . .

# 安装依赖
RUN pip3 install -r requirements.txt
RUN pip install jieba flashtext -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

# 共享SSH密钥：
COPY ~/.ssh/id_rsa /root/.ssh/id_rsa
COPY ~/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 600 /root/.ssh/id_rsa.pub

# 可选步骤，用于预热模块
RUN python3  -c 'from check_proxy import warm_up_modules; warm_up_modules()'

# 启动
CMD ["python3", "-u", "main.py"]