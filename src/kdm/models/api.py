"""OpenAI-compatible local inference for the complete baseline census only."""
from __future__ import annotations
import base64, io
import requests

class LocalChat:
    def __init__(self,url,model,timeout=180):
        self.url=url.rstrip('/');self.model=model;self.timeout=timeout
        from urllib.parse import urlsplit
        parsed=urlsplit(self.url)
        if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}:
            raise ValueError("Use a local inference service")
    def generate_text(self,image,prompt,max_tokens=64,seed=0,temperature=0.0):
        content=[]
        if image is not None:
            buf=io.BytesIO();image.save(buf,format='PNG')
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                base64.b64encode(buf.getvalue()).decode()}})
        content.append({'type':'text','text':prompt})
        payload={'model':self.model,'messages':[{'role':'user','content':content}],
                 'max_tokens':max_tokens,'temperature':temperature,'seed':seed,
                 'chat_template_kwargs':{'enable_thinking':False}}
        response=requests.post(self.url+'/chat/completions',json=payload,timeout=self.timeout)
        response.raise_for_status();data=response.json()
        choice=data['choices'][0]
        text=choice['message']['content']
        if not isinstance(text,str): raise ValueError("Model returned no text")
        return {'text':text,'finish_reason':choice.get('finish_reason'),
                'status':'ok','usage':data.get('usage',{})}
