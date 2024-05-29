from transformers import VitsModel, VitsTokenizer, set_seed
import torch
from shannlp import util, word_tokenize
import gradio as gr
import numpy as np

model = VitsModel.from_pretrained("NorHsangPha/mms-tts-shn-train")
tokenizer = VitsTokenizer.from_pretrained("NorHsangPha/mms-tts-shn-train")


def preprocess_string(input_string: str):
    string_token = word_tokenize(input_string)
    num_to_shanword = util.num_to_shanword

    result = []
    for token in string_token:
        if token.strip().isdigit():
            result.append(num_to_shanword(int(token)))
        else:
            result.append(token)

    full_token = "".join(result)
    return full_token


def generate_audio(input_string: str):
    processed_string = preprocess_string(input_string)
    inputs = tokenizer(processed_string, return_tensors="pt")

    set_seed(42)

    model.speaking_rate = 0.6
    model.noise_scale = 0.8

    with torch.no_grad():
        output = model(**inputs).waveform

    return (model.config.sampling_rate, output.numpy().flatten())


title = "MMS-TTS-Shan finetune Demo"

examples = [
    ["မႂ်ႇသုင်ၶႃႈ ယူႇလီၵိၼ်ဝၢၼ်ၵတ်းယဵၼ် လီယူႇၶႃႈၼေႃႈ။"],
    ["ပဵၼ်ယၢမ်းဢၼ် ၸႂ်တိုၼ်ႇတဵၼ်ႈ ၽူင်ႉပိဝ် တႃႇၼုမ်ႇယိင်းၼုမ်ႇၸၢႆးၶဝ် ၸိူဝ်းဢၼ် တေလႆႈၶိုၼ်ႈႁဵၼ်းၼၼ်ႉယူႇ"],
    [
        "မိူဝ်ႈပီ 1958 လိူၼ်မေႊ 21 ဝၼ်းၼၼ်ႉ ၸဝ်ႈၼွႆႉသေႃးယၼ်ႇတ ဢမ်ႇၼၼ် ၸဝ်ႈၼွႆႉ ဢွၼ်ႁူဝ် ၽူႈႁၵ်ႉၸိူဝ်ႉၸၢတ်ႈ 31 ၵေႃႉသေ တိူင်ႇၵၢဝ်ႇယၼ်ႇၸႂ် ၵိၼ်ၼမ်ႉသတ်ႉၸႃႇ တႃႇၵေႃႇတင်ႈပူၵ်းပွင် ၵၢၼ်လုၵ်ႉၽိုၼ်ႉ တီႈႁူၺ်ႈပူႉ ႁိမ်းသူပ်းၼမ်ႉၵျွတ်ႈ ၼႂ်းဢိူင်ႇမိူင်းႁၢင် ၸႄႈဝဵင်းမိူင်းတူၼ် ၸိုင်ႈတႆးပွတ်းဢွၵ်ႇၶူင်း လႅၼ်လိၼ်ၸိုင်ႈထႆး။"
    ],
    [
        "ပဵၼ်ၵၢၼ်ၾုၵ်ႇၾင်ၸႂ်ၵၼ်ၼႅၼ်ႈ ၼၵ်းပၵ်းၸႂ် ယွင်ႈၵုၼ်းယွင်ႈမုၼ်ဢူငဝ်း ၸိူဝ်းၽူႈလဵပ်ႈႁဵၼ်းႁူႉပိုၼ်း ၸဵမ်လဵၵ်ႉယႂ်ႇၼုမ်ႇထဝ်ႈ ၼႂ်းၸိူဝ်း ၽူႈႁၵ်ႉ ၸိူဝ်ႉၸၢတ်ႈလၢႆပၢၼ်လၢႆသႅၼ်းမႃး 66 ပီ ၼပ်ႉတင်ႈတႄႇ 1958 ဝၼ်းတီႈ 21 လိူၼ်မေႊ။"
    ],
]

# iface = gr.Interface(
#     title=title,
#     fn=generate_audio,
#     inputs="text",
#     outputs="audio",
#     examples=examples,
#     layout="vertical",
#     description="Finetune MMS-TTS-Shan model from 184 samples of Shan dataset",
# )


def update_button_style(input_text):
    if input_text.strip():
        return gr.update(
            value="Generate Speech", variant="primary", interactive=True
        )  # Orange style
    else:
        return gr.update(
            value="Enter text to enable", variant="secondary", interactive=False
        )  # Gray style


with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Speech Interface")
    gr.Markdown("Enter text to generate speech using VitsModel")

    with gr.Row():
        input_text = gr.Textbox(label="Input Text")
        output_audio = gr.Audio(label="Output Audio")

    submit_btn = gr.Button(
        value="Generate Speech", variant="secondary", interactive=False
    )

    # Display examples in a single-column layout
    with gr.Column():
        gr.Examples(examples, inputs=input_text, label=None)

    input_text.change(fn=update_button_style, inputs=input_text, outputs=submit_btn)
    submit_btn.click(fn=generate_audio, inputs=input_text, outputs=output_audio)

demo.launch()
