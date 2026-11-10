import json
import os
from pathlib import Path



def generate_json(prompt, payload, debug_file=None):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url=os.getenv('DEEPSEEK_BASE_URL', '<DEEPSEEK_BASE_URL>'),
    )
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': '请只返回完整、合法的 JSON 对象。\n输入数据：\n'
         + json.dumps(payload, ensure_ascii=False)},
    ]

    for attempt in range(2):
        response = client.chat.completions.create(
            model=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'),
            messages=messages,
            response_format={'type': 'json_object'},
            temperature=0,
            max_tokens=12000,
            extra_body={'thinking': {'type': os.getenv('DEEPSEEK_THINKING', 'disabled')}},
        )
        choice = response.choices[0]
        content = choice.message.content or ''
        finish_reason = choice.finish_reason
        try:
            if finish_reason != 'stop':
                raise ValueError(f'Unexpected finish_reason={finish_reason!r}')
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError('The JSON root must be an object')
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            if debug_file is not None:
                path = Path(debug_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_name(path.stem + f'_attempt{attempt + 1}.txt').write_text(
                    f'finish_reason={finish_reason}\n{content}', encoding='utf-8'
                )
            if attempt == 1:
                raise ValueError(
                    f'DeepSeek returned invalid/incomplete JSON after 2 attempts; '
                    f'finish_reason={finish_reason}; saved raw response near {debug_file}'
                ) from exc
            print(f'Invalid DeepSeek JSON ({exc}); retrying once...', flush=True)

    raise RuntimeError('Unreachable')
