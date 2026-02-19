import json

def create_marketplace_files(job_dir):
    seo_path = f"{job_dir}/seo.json"

    with open(seo_path, "r") as f:
        seo = json.load(f)

    # ETSY LISTING
    etsy_text = f"""
TITLE:
{seo['title']}

DESCRIPTION:
{seo['description']}

TAGS:
{', '.join(seo['tags'])}
"""
    with open(f"{job_dir}/etsy_listing.txt", "w", encoding="utf-8") as f:
        f.write(etsy_text.strip())

    # REDBUBBLE TAGS
    with open(f"{job_dir}/redbubble_tags.txt", "w") as f:
        f.write(", ".join(seo["tags"]))

    # AMAZON MERCH KEYWORDS
    with open(f"{job_dir}/amazon_keywords.txt", "w") as f:
        f.write(", ".join(seo["keywords"]))
