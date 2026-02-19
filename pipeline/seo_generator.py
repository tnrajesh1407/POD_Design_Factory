import base64
import json
from openai import OpenAI

client = OpenAI()

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_seo_metadata(image_path):

    base64_image = encode_image(image_path)

    prompt = """
You are a Print-on-Demand SEO expert.

Look at this t-shirt design and generate marketplace listing content.

Return STRICT JSON with fields:
title
description
alt_text
tags (array of 30 tags)
keywords (array of 20 keywords)

Style:
• Optimized for Etsy, Redbubble, Amazon Merch
• Catchy title (max 80 chars)
• SEO rich description (2 paragraphs)
• Tags must be short phrases
• Keywords must be search phrases
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    result = json.loads(response.choices[0].message.content)
    return result
