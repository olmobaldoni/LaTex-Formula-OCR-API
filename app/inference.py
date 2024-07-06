import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from transformers import NougatImageProcessor

import logging
import io


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class NougatBase:

    def __init__(self):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self.model_name = "Norm/nougat-latex-base"
        logging.info(f"Model name: {self.model_name}")

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)

        self.tokenizer = NougatTokenizerFast.from_pretrained(self.model_name)

        self.latex_processor = NougatImageProcessor.from_pretrained(self.model_name)

    def preprocess_image(self, image):

        img = Image.open(io.BytesIO(image))

        if not img.mode == "RGB":
            img = img.convert("RGB")

        pixel_values = self.latex_processor(img, return_tensors="pt").pixel_values

        decoder_input_ids = self.tokenizer(
            self.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        return pixel_values, decoder_input_ids

    def inference_latex_code(self, image):

        pixel_values, decoder_input_ids = self.preprocess_image(image=image)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_length,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=5,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        sequence = self.tokenizer.batch_decode(outputs.sequences)[0]
        sequence = (
            sequence.replace(self.tokenizer.eos_token, "")
            .replace(self.tokenizer.pad_token, "")
            .replace(self.tokenizer.bos_token, "")
        )
        return sequence