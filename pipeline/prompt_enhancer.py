from openai import OpenAI
import os, re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a professional print-on-demand designer.

Rewrite user prompts into HIGH QUALITY t-shirt design prompts.

Every raw prompt from user shuold be well-crafted prompt as per following four key components:
  1. Primary Subject: The central focus of the image
  2. Subject Behavior: Actions, poses, or states
  3. Visual Style: Artistic medium or any other styles 
  4. Environmental Context: Setting, atmosphere, and mood

Hard rules:
- output should be centered composition
- centered composition
- isolated subject
- TRANSPARENT BACKGROUND ONLY (no white background)
- no background, no scenery, no backdrop
- high contrast colors
- screen print friendly
- minimal tiny details
- symmetrical layout when possible
- Images should well fit on t-shirt and mockups when user decided to print on t-shirt (POD)
- No mockup related content such as mockup Tshirts, mockup jackets etc   
- No gibberish content
- NO watermark, NO logo, NO BRANDS, NO ADULT CONTENT, NO NUDITY, NO VIOLENCE, NO gorey details,NO DANGEROUS CONTENT,NO DISTURBING CONTENT,NO UNCOMFORTABLE CONTENT  

Return ONLY the enhanced prompt.
"""

def enhance_prompt(user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        enhanced = resp.choices[0].message.content.strip()

        # Safety cleanup: remove/replace background instructions that hurt POD cutouts
        enhanced = re.sub(r"\bwhite background\b", "transparent background", enhanced, flags=re.I)
        enhanced = re.sub(r"\bsolid white background\b", "transparent background", enhanced, flags=re.I)
        enhanced = re.sub(r"\bplain background\b", "transparent background", enhanced, flags=re.I)

        print("Enhanced prompt:", enhanced)
        return enhanced

    except Exception as e:
        print("Prompt enhancement failed:", e)
        return user_prompt
