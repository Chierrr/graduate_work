from flask import Flask, request, Response
from io import BytesIO
import torch
from av import open as avopen

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile

# Flask Init
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def get_text(text, language_str, hps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        x_tst = phones.to(dev).unsqueeze(0)
        tones = tones.to(dev).unsqueeze(0)
        lang_ids = lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        return audio


def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text


def wav2(i, o, format):
    inp = avopen(i, "rb")
    out = avopen(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

dev = "cuda"
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(dev)
_ = net_g.eval()

_ = utils.load_checkpoint("logs/alkaid/G_19000.pth", net_g, None, skip_optimizer=True)


# 定义 Flask 路由处理函数，当访问根路径（"/"）时执行此函数
@app.route("/")
def main():
    try:
        # 从请求参数中获取发言人（speaker）
        speaker = request.args.get("speaker")

        # 从请求参数中获取文本内容，并替换其中的换行符为无（因为"/n"应该是一个错误，正确的换行符是"\n"）
        text = request.args.get("text").replace("\n", "")

        # 从请求参数中获取 sdp_ratio 的值，如果未提供则默认为 0.2
        sdp_ratio = float(request.args.get("sdp_ratio", 0.2))

        # 从请求参数中获取 noise 的值，如果未提供则默认为 0.5
        noise = float(request.args.get("noise", 0.5))

        # 从请求参数中获取 noisew 的值，如果未提供则默认为 0.6
        noisew = float(request.args.get("noisew", 0.6))

        # 从请求参数中获取 length 的值，如果未提供则默认为 1.2
        length = float(request.args.get("length", 1.2))

        # 从请求参数中获取语言类型
        language = request.args.get("language")

        # 如果 length 大于等于 2，则返回错误信息
        if length >= 2:
            return "Too big length"

            # 如果文本长度大于等于 250，则返回错误信息
        if len(text) >= 250:
            return "Too long text"

            # 从请求参数中获取音频格式，如果未提供则默认为 "wav"
        fmt = request.args.get("format", "wav")

        # 如果 speaker 或 text 为空，则返回错误信息
        if None in (speaker, text):
            return "Missing Parameter"

            # 如果音频格式不是 "mp3", "wav", 或 "ogg" 之一，则返回错误信息
        if fmt not in ("mp3", "wav", "ogg"):
            return "Invalid Format"

            # 如果语言类型不是 "JA" 或 "ZH" 之一，则返回错误信息
        if language not in ("JA", "ZH"):
            return "Invalid language"

    except:
        # 如果在尝试获取和处理参数时出现任何异常，则返回错误信息
        return "Invalid Parameter"

        # 使用 torch 的 no_grad 上下文管理器，以避免计算梯度，节省内存
    with torch.no_grad():
        # 调用 infer 函数来生成音频，使用前面获取的参数
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language,
        )

        # 使用 BytesIO 对象作为临时文件来存储音频数据
    with BytesIO() as wav:
        # 将音频数据写入 BytesIO 对象中，使用 hps.data.sampling_rate 作为采样率
        wavfile.write(wav, hps.data.sampling_rate, audio)

        # 释放 CUDA 缓存
        torch.cuda.empty_cache()

        # 如果请求的音频格式是 "wav"，则直接返回音频数据
        if fmt == "wav":
            return Response(wav.getvalue(), mimetype="audio/wav")

            # 否则，将 BytesIO 对象中的 wav 格式音频数据转换为请求的音频格式
        wav.seek(0, 0)  # 将文件指针重置到文件开头
        with BytesIO() as ofp:
            # 使用 wav2 函数将 wav 格式转换为请求的音频格式，并将结果写入 ofp
            wav2(wav, ofp, fmt)
            # 返回转换后的音频数据，设置正确的 MIME 类型
            return Response(
                ofp.getvalue(), mimetype="audio/mpeg" if fmt == "mp3" else "audio/ogg"
            )

if __name__ == '__main__':
    # 确保端口号没有被其他应用占用
    PORT = 5000
    app.run(host='0.0.0.0', port=PORT, debug=True)