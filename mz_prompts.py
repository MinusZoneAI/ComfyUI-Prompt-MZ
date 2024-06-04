Beautify_Prompt = """
Stable Diffusion is an AI art generation model similar to DALLE-2.
Below is a list of prompts that can be used to generate images with Stable Diffusion:
- portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski
- pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson
- ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image
- red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski
- a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt
- athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration
- closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo
- ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha
I want you to write me a list of detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more.
"""

Long_prompt = "Long prompt version should consist of 3 to 5 sentences. Long prompt version must sepcify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe any atmosphere!!!  The language of reply is English only!!!"

Standardize_Prompt = """
Extract the content about Stable Diffusion style from the following input and combine it into a json array. Note that the output will be directly used in the program. 
Please output the standardized json content.
"""


GPT4VImageCaptioner_System = """
As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. 
"""

# 来自https://github.com/jiayev/GPT4V-Image-Captioner
GPT4VImageCaptioner_Prompt = """
Employ succinct keywords or phrases or sentence, steering clear of elaborate sentences and extraneous conjunctions. 
Prioritize the tags by relevance. 
Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. 
When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. 
For other image categories, apply appropriate and common descriptive tags as well. 
Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. 
Your tags should be accurate, non-duplicative, and within a 20-75 word count range. 
These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. 
Tags should be comma-separated.
"""


M_ImageCaptioner_System = """
Long prompt version should consist of 3 to 5 sentences. Long prompt version must sepcify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe any atmosphere!!!  
"""

M_ImageCaptioner_Prompt = """
Describe this image in detail please.
The language of reply is English only!!!
Starts with "In the image," 
"""


M_ImageCaptioner2_System = """
You are an assistant who perfectly describes images. 
"""

M_ImageCaptioner2_Prompt = """
Describe this image in detail please.
The language of reply is English only!!!
Starts with "In the image," 
"""


ImageCaptionerPostProcessing_System = """
I want you to write me a detailed list of tips for Content.
Write a very short description of the scene and put it in the 'short_describes' field
Write complete [moods, styles, lights, elements, objects] of the word array and put it in the '$_tags' field
Don't include anything that isn't in Content.
The language of reply is English only!!!
"""
