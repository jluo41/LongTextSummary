import streamlit as st
import pandas as pd
# import other necessary libraries
from langchain.chat_models import ChatOpenAI, ChatAnyscale
from langchain.prompts import ChatPromptTemplate
import re
import json
import pandas as pd
import base64
# from apikey import openai_api_key


def get_llm_chat(modelname, temperature, openai_api_key = None, anyscale_api_key = None):
    model_to_apimodelname = {
        'gpt3.5': {'model': 'gpt-3.5-turbo-1106', 'openai_api_key': openai_api_key},
        'gpt4': {'model': 'gpt-4-1106-preview', 'openai_api_key': openai_api_key},
        # 'llama2-70b': {'model_name': 'meta-llama/Llama-2-70b-chat-hf', 'anyscale_api_key': anyscale_api_key},
        # 'llama2-13b': {'model_name': 'meta-llama/Llama-2-13b-chat-hf', 'anyscale_api_key': anyscale_api_key},
    }
    para = model_to_apimodelname[modelname]
    # if modelname in ['gpt3.5', 'gpt4']: 
    #     chat = ChatOpenAI(temperature = temperature,  **para)
    # elif modelname in ['llama2-70b', 'llama2-13b']:
    #     chat = ChatAnyscale(temperature = temperature, **para)
    # else:
    #     raise ValueError(f'{modelname} is not available')
    chat = ChatOpenAI(temperature = temperature, **para)
    return chat

def convert_response_to_json(text):
    pattern = r'\[\s*(?:{.*?}\s*,\s*)*{.*?}\s*\]'
    matches = re.findall(pattern, text)
    # print(matches)
    match = matches[0] # take the first one
    d = json.loads(match)
    return d


def get_current_chunk_topics_and_endidx(topic_list, chunk_start, chunk_end, file_sentences, ignore_num = 3):
    topic_list = topic_list.copy()
    topics = []
    for i in topic_list: i = i.copy(); i['end'] = i['end']  + 1; topics.append(i)
       
    last_topic = topics[-1]
    last_start = last_topic['start']
    last_end = last_topic['end']
    last_topic_name = last_topic['topic']
    
    # case 1: final_topics we want to return
    if len(topics) == 1:
        topic = topics[0]
        topic_name = topic['topic']
        current_start = topic['start']
        current_end = topic['end']
        topic_sentences = file_sentences[current_start: current_end]
        d = {}
        d['topic'] = topic_name
        d['start'] = current_start
        d['end'] = current_end
        d['topic_sentences'] = topic_sentences
        final_topics = [d]
        next_chunk_start = current_end
        return final_topics, next_chunk_start
    
    
    # case 2: multiple topics
    final_topics = []
    for i in range(len(topics) - 1):
        topic = topics[i]
        topic_name = topic['topic']
        current_start = topic['start']
        current_end = topic['end']
        next_start = topics[i + 1]['start']
        
        if current_end >= chunk_end - ignore_num and last_end != len(file_sentences):
            print('\n<--------stop because touch the Included End\n')
            break 
            
        if current_end != next_start:
            print('\n<--------stop because dis-match\n')
            break
            
        d = {}
        topic_sentences = file_sentences[current_start: current_end]
        d['topic'] = topic_name
        d['start'] = current_start
        d['end'] = current_end
        d['topic_sentences'] = topic_sentences
        final_topics.append(d)
        next_chunk_start = current_end
        
    # after the above loop, we have two cases
    if i == 0:
        topic = topics[0]
        topic_name = topic['topic']
        current_start = topic['start']
        current_end = topic['end']
        topic_sentences = file_sentences[current_start: current_end]
        d = {}
        d['topic'] = last_topic_name
        d['start'] = current_start
        d['end'] = current_end
        d['topic_sentences'] = topic_sentences
        final_topics = [d]
        next_chunk_start = current_end
        return final_topics, next_chunk_start
        
    # print(len(final_topics))
    # print(len(topics) - 1)
    # print(last_end == len(file_sentences))
    if len(final_topics) == len(topics) - 1 and last_end == len(file_sentences):
        d = {}
        topic_sentences = file_sentences[last_start: last_end]
        d['topic'] = last_topic_name
        d['start'] = last_start
        d['end'] = last_end
        d['topic_sentences'] = topic_sentences
        next_chunk_start = last_end
        final_topics.append(d)
            
    return final_topics, next_chunk_start



def convert_review_into_related_sentence_list(INPUT_SENTENCES, 
                                              PROMPT_OUTPUTFORMAT, 
                                              PROMPT_REQUIREMENT, 
                                              PROMPT_TEMPLATE, 
                                              chat
                                             ):
    d = {'INPUT_SENTENCES': INPUT_SENTENCES, 
         'PROMPT_OUTPUTFORMAT': PROMPT_OUTPUTFORMAT, 
         'PROMPT_REQUIREMENT': PROMPT_REQUIREMENT}
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    messages = prompt_template.format_messages(**d)
    print(messages[0].content)
    response = chat(messages)
    # output = {'chat': chat, 'prompt': messages[0].content,  'model': response.content}
    return messages[0].content, response.content


def summarize_file_sentences(file_sentences, chunk_size, ExcludeNum,
                             PROMPT_OUTPUTFORMAT, PROMPT_REQUIREMENT, PROMPT_TEMPLATE, 
                             chat):

    Final_Topics_for_Documentation = []
    chunk_idx = 0
    chunk_start = 0
    while True:
        chunk_idx = chunk_idx + 1
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(file_sentences): chunk_end = len(file_sentences)
        
        chunk_sentences = file_sentences[chunk_start: chunk_end]
        INPUT_SENTENCES = '\n'.join(chunk_sentences)
        
        print(f"\n\n============= Chunk {chunk_idx}: s-{chunk_start}, e-{chunk_end} ============")
        # print(INPUT_SENTENCES)
        # chat = chat_dict[modelname]
        result = convert_review_into_related_sentence_list(INPUT_SENTENCES, 
                                                        PROMPT_OUTPUTFORMAT, 
                                                        PROMPT_REQUIREMENT, 
                                                        PROMPT_TEMPLATE, 
                                                        chat)
        prompt, response = result
        topic_list = convert_response_to_json(response)
        print(topic_list)
        final_topics, next_chunk_start = get_current_chunk_topics_and_endidx(topic_list, 
                                                                            chunk_start, chunk_end, 
                                                                            file_sentences, 
                                                                            ExcludeNum)
        for topic in final_topics:
            print('* [TOPIC]', topic['topic'], topic['start'], topic['end']-1)
            topic_sentences = topic['topic_sentences']
            print('\n'.join(topic_sentences))
            print('\n\n')
            
        print('\nnext_chunk_start is:', next_chunk_start)
        Final_Topics_for_Documentation = Final_Topics_for_Documentation + final_topics
        if next_chunk_start == len(file_sentences): break
        chunk_start = next_chunk_start

    df = pd.DataFrame(Final_Topics_for_Documentation)
    return df


##########################
PROMPT_OUTPUTFORMAT = '''
请返回如下json格式：
[
{"topic": "主题内容“, "start": 1 , "end": 10},  
{"topic": "主题内容“, "start": 11 , "end": 19}, 
{"topic": "主题内容“, "start": 20 , "end": 38},
 ....
]
'''

PROMPT_REQUIREMENT = '''
返回的内容有如下要求：
1. 请注意将json例子里的"主题内容"和数字换成正确的主题内容和数字。
2. 请只返回json，不需要进行解释。
3. 主题所包换的句子要尽量的多，主题的个数尽量少。
4. 主题的请尽量宏大抽象。
5. 确保每个主题的结尾句id比下一个主题的开始句id大1。
'''

PROMPT_TEMPLATE = """
你是一个保险培训师，准备将视频文稿整理成学习文档。

文档如下：
```\n{INPUT_SENTENCES}\n```

请将以上视频文稿进行切段，每一段给一个主题，并且指明每一段第一句和最后一句的序号。

{PROMPT_OUTPUTFORMAT}
{PROMPT_REQUIREMENT}
"""
##########################


# Define the main function of the app
def main():
    st.title("Your App Title")


    # File upload
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the file and process it
        file = uploaded_file.getvalue().decode("utf-8"); sep = '。'
        # with open(path, 'r') as f: file = f.read()
        

    # Sidebar for parameters
    chunk_size = st.sidebar.number_input('Chunk Size', min_value=1, value=30)
    exclude_num = st.sidebar.number_input('Exclude Number', min_value=1, value=3)
    modelname = st.sidebar.selectbox('Model Name', ['gpt4', 'gpt3.5'])
    openai_api_key = st.sidebar.text_input('openai_api_key')

    
    if openai_api_key != '':
        anyscale_api_key = None
        temperature = 0
        # chat_dict = {}
        # modelname_list = ['gpt4', 'chatgpt']
        # for modelname in modelname_list:
        #     chat = get_llm_chat(modelname, temperature, openai_api_key, anyscale_api_key)
        #     chat_dict[modelname] = chat
        chat = get_llm_chat(modelname, temperature, openai_api_key, anyscale_api_key)

        prompt_outputformat = st.text_area("PROMPT_OUTPUTFORMAT", PROMPT_OUTPUTFORMAT)
        prompt_requirement = st.text_area("PROMPT_REQUIREMENT", PROMPT_REQUIREMENT)
        prompt_template = st.text_area("PROMPT_TEMPLATE", PROMPT_TEMPLATE)

        # Process the file and parameters
        if st.button('Process File'):
            if uploaded_file is not None:
                # Call your processing functions here
                # Display the results
                file_sentences = [i + sep for i in file.split(sep)]
                file_sentences = [str(idx)+':' + sent for idx, sent in enumerate(file_sentences)]
                # print('file sentences number:', len(file_sentences))
                st.write('file sentences number:', len(file_sentences))

                df = summarize_file_sentences(file_sentences, chunk_size, exclude_num,
                                            prompt_outputformat, prompt_requirement, prompt_template, chat)

                
                # Display the results
                st.write("Processed Results")
                st.dataframe(df)

                # Convert DataFrame to CSV for download
                csv = df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv).decode()  # some browsers need base64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

    else:
        st.markdown('Please input your openai_api_key')

# Check if the script is the main module
if __name__ == "__main__":
    main()
