import leafmap.foliumap as leafmap
import folium
import gradio as gr
import random
import pandas as pd
import numpy as np
import re
import json
import os
import requests

poi_embedding = np.load('./poi_embedding.npy')
df_location = pd.read_csv("./points_info.csv")
df_location = df_location.rename(
                columns={
                    "name": "景点名称",
                    "level": "景点类别",
                    "address": "景点地址",
                    "intro": "景点简介",
                    "short_intro": "景点介绍",
                    "recomm_score": "推荐指数",
                    "recomm_reason": "推荐理由",
                }
            )


def l2_normalization(embedding:np.ndarray) -> np.ndarray:
    if embedding.ndim == 1:
        return embedding / np.linalg.norm(embedding).reshape(-1,1)
    else:
        return embedding/np.linalg.norm(embedding,axis=1).reshape(-1,1)


def find_related_doc(query, origin_chunk, poi_embedding, top_k=5):

    # query_response = erniebot.Embedding.create(model='ernie-text-embedding',input=[query])
    url = "https://cloud.baidu.com/api/qianfan_agent/v1/embedding?qianfan_api_token=" + os.environ.get("QIANFAN_API_TOKEN")
    payload = json.dumps({
        "texts": [query]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    query_embedding = np.array(json.loads(response.text)["data"][0]["embedding"])
    # query_embedding = np.array(query_response.get_result()[0])
    rank_scores = l2_normalization(query_embedding) @ poi_embedding.T
    top_k_similar = rank_scores.argsort()[0][-top_k:].tolist()[::-1]

    return top_k_similar

def generate_info(prompt):
    response = erniebot.ChatCompletion.create(model="ernie-bot-4", # "ernie-bot",
                          messages=[{"role": "user", "content": prompt}],
                          system="你是一名优秀的导游，擅长使用优美的文字，深入了解景点的特点以及背后的名人轶事和历史典故",
                            )

    text = response.get_result()
    cleaned_text = re.sub(r'json', '', text)
    cleaned_text = re.sub(r'```', '', cleaned_text)
    return cleaned_text.strip()

def generate_info_qianfan(prompt):
    url = "https://cloud.baidu.com/api/qianfan_agent/v1/chat?qianfan_api_token="+os.environ["QIANFAN_API_TOKEN"]

    payload = json.dumps({
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ],
        "system":"你是一名优秀的导游，擅长使用优美的文字，深入了解景点的特点以及背后的名人轶事和历史典故",
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(json.loads(response.text)["result"])
    text = json.loads(response.text)["result"]
    cleaned_text = re.sub(r'json', '', text)
    cleaned_text = re.sub(r'```', '', cleaned_text)
    return cleaned_text.strip()

def generate_map(places=[], zoom_level=14):

    coords = [32.39679700000001, 119.439857]
    lat, lon = float(coords[0]), float(coords[1])
    map = leafmap.Map(location=(lat,lon),
             tiles="https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7",
            #  tiles="https://webrd02.is.autonavi.com/appmaptile?lang=zh_en&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
             attr="高德-常规图",
             zoom_start=zoom_level)



    for rowid ,row in (df_location.sample(100).iterrows() if len(places) == 0 \
                       else df_location[df_location["景点名称"].isin(places)].iterrows()):

        folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=row['景点名称'],
                icon=folium.Icon(color="green"),
            ).add_to(map)
    return map.to_gradio()

def respond(message, chat_history):

    top_k_similar = find_related_doc(message,list(df_location['景点简介']),poi_embedding,10)
    prompt = """
        ```
        %s
        ```
        你是一名优秀的导游，上面是关于景点的Json格式的介绍；
        我的需求是：%s。
        请你既要参考我的需求，又要考虑景点的多样性，从我提供的景点信息中挑选出5个景点，并规划一条旅游路线；
        请按照"{"route":[景点名称1,景点名称2,景点名称3...,景点名称5]","intro":"结合我的需求生成旅游路线的介绍，300字左右"}"输出；
        确保返回的内容仅有{}包含的内容，能直接被Python读取为JSON；
        除了JSON不要有其他内容、系统提示或者注意事项。
        """ % (
            df_location.iloc[top_k_similar].loc[:, ["景点名称","景点介绍"]
            ].to_json(orient="records", force_ascii=False), # .to_markdown(),
            message,
        )

    response = generate_info_qianfan(prompt)
    text = json.loads(response) # 去除额外的空白字符
    bot_message = text['intro']
    chat_history.append((message, bot_message))
    return "", chat_history, generate_map(text["route"])

with gr.Blocks(title="Citywalk智慧规划", theme="soft") as demo:

    gr.Markdown("### 「城语」- 探索城市故事 🚶")
    # gr.Markdown("用大模型帮您规划Citywalk路线，探索城市的独特风景...")
    with gr.Row():

        with gr.Column(scale=1):
            chatbot = gr.Chatbot([(None,"请告诉我，您希望有怎样的旅行体验...？")],avatar_images=("https://i.imgs.ovh/2023/12/16/6xmZl.png","https://i.imgs.ovh/2023/12/16/6x6xd.png"))
            msg = gr.Textbox(value="请设计一条体验烟火气的路线",container=True,show_label=False,info="请输入您的需求")
            # msg.submit(respond,[msg, chatbot], [msg, chatbot, map_output])
        with gr.Column(scale=1):
            map_output = gr.HTML(value=generate_map(), label="Travel map")
    with gr.Row():

        chat_btw = gr.Button("提交需求", variant="primary", scale=1)
        chat_btw.click(respond,[msg, chatbot], [msg, chatbot, map_output])
        clear = gr.ClearButton([msg, chatbot], scale=1)

    # map_button.click(generate_map, inputs=[coordinates_input,zoom_level_input], outputs=[map_output])

# run this in a notebook to display the UI
demo.queue().launch(debug=False)