import config
import json
from pathlib import Path
import pytesseract
from PIL import Image
import openai
import PyPDF2

class AgenticIDP:
    def __init__(self, input_dir="INPUT/", schema_path="extraction_schema.json"):
        self.gpt_model = "gpt-4o-mini"
        self.prompt_doc_max_length = 3000
        self.input_dir = Path(input_dir)
        self.categories = self._load_categories(schema_path)
        self.extraction_schemas = self._load_schemas(schema_path)
        self.openai_client = openai.OpenAI(api_key=config.get_chatGPT_api_key())
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def _load_categories(self, schema_path):
        with open(schema_path) as f:
            schemas = json.load(f)
        return list(schemas.keys()) + ["Unknown"]
    
    def _load_schemas(self, schema_path):
        with open(schema_path) as f:
            return json.load(f)

    def _extract_text(self, doc_path):
        suffix = doc_path.suffix.lower()
        
        if suffix == '.pdf':
            try:
                with open(doc_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ' '.join(page.extract_text() for page in reader.pages)
                
                if not text.strip():
                    return self._perform_ocr(doc_path)
                return text
            except Exception:
                return self._perform_ocr(doc_path)
                
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._perform_ocr(doc_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def process_documents(self):
        for doc_path in self.input_dir.glob("*"):
            try:
                text = self._extract_text(doc_path)
                category = self._categorize_document(text)
                extracted_data = self._extract_data(text, category)
                self._save_results(doc_path.stem, {
                    "categories": category,
                    "extracted_data": extracted_data
                })
            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")

    def _perform_ocr(self, doc_path):
        try:
            image = Image.open(doc_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")

    def _categorize_document(self, text):
        prompt = f"""Here is an incoming document:
{text[:self.prompt_doc_max_length]}...<end of doc>

Categories: {', '.join(self.categories)}. """
        prompt += """Return a JSON array of applicable categories ranked by probability. Example: '{\n  "categories": ["Invoice", "Receipt"]\n}'"""

        response = self.openai_client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            logprobs=True, # to get log probabilities of tokens (for confidence thresholding)
            temperature=0
        )

        all_logprobs = response.choices[0].logprobs.content
        probabilities = []
        for i, category in enumerate(self.categories):
            for token_logprob in all_logprobs:
                if category == token_logprob.token:
                    probabilities.append({"category": category, "probability": token_logprob.logprob})
                    continue

        highest = max(probabilities, key=lambda x: x['probability'])
        logprob_threshold = -2.0 

        if highest['probability'] > logprob_threshold:
            print(f"Selected category: {highest['category']} with log probability: {highest['probability']}")
            return highest['category']
        else:
            print("No category met the confidence threshold.")
            return "Unknown"

    def _extract_data(self, text, category):
        schema = self.extraction_schemas.get(category, {})
        if not schema:
            return {}

        prompt = f"""From this document:
{text[:self.prompt_doc_max_length]}...<end of doc>

Extract the following elements:
{json.dumps(schema, indent=2)}

Return ONLY the JSON data."""

        response = self.openai_client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)

    def _save_results(self, doc_name, results):
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f"{doc_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    processor = AgenticIDP()
    processor.process_documents()
