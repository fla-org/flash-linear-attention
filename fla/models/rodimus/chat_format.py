import copy
import dataclasses
import logging
import re
import uuid
from copy import deepcopy
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


class PromptStyle(IntEnum):
    '''Prompt styles.'''
    RODIMUS_CHAT = auto()
    CHATML = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATGLM3 = auto()
    BAICHUAN2 = auto()


@dataclasses.dataclass
class Chat:

    id: str = None
    name: Optional[str] = None
    prompt_style: Optional[PromptStyle] = None

    system_template: str = '<role>SYSTEM</role>{}'
    system_message: str = ''

    role_human: str = 'HUMAN'
    role_assistant: str = 'ASSISTANT'
    role_observation: str = 'OBSERVATION'
    role_template: str = '<role>{}</role>'

    turn_start: str = ''
    human_end: str = ''
    assistant_start: str = ''
    assistant_end: str = ''
    assistant_end_ids: Optional[List[int]] = None
    general_role_end: str = ''

    tool_template = '<tool>{}</tool>'
    code_template = '<code>{}</code>'
    arithemetic_templte = '<arithemetic>{}</arithemetic>'
    image_template = '<image>{}</image>'

    messages: List[Tuple[str, str]] = ()

    offset: int = 0

    source: Optional[str] = None
    lang: Optional[str] = None
    topic: Optional[str] = None

    origin_json: Optional[dict] = None

    @classmethod
    def from_json(
        cls,
        input: dict,
        name: Optional[str] = None,
        prompt_style: Optional[PromptStyle] = None,
    ):
        _id = input.get('id')
        if name:
            _name = name
        else:
            _name = input.get('name')
        source = input.get('source')
        lang = input.get('lang')
        topic = input.get('topic')
        kwargs = {}
        if 'system_template' in input:
            kwargs['system_template'] = input['system_template']
        if 'system_message' in input:
            kwargs['system_message'] = input['system_message']

        chat = cls(
            id=_id,
            name=_name,
            prompt_style=prompt_style,
            source=source,
            lang=lang,
            topic=topic,
            origin_json=deepcopy(input),
            **kwargs,
        )
        if 'messages' in input:
            for msg in input['messages']:
                if msg['role'] == 'HUMAN':
                    role = chat.role_human
                elif msg['role'] == 'OBSERVATION':
                    role = chat.role_observation
                elif msg['role'] == 'ASSISTANT':
                    role = chat.role_assistant
                else:
                    raise ValueError(f'不支持数据集中的 role: {msg["role"]}')

                chat.append_message(role, msg['content'])

        elif 'turns' in input:
            for turn in input['turns']:
                if 'HUMAN' in turn:
                    content = turn['HUMAN']
                    chat.append_message(chat.role_human, content)
                if 'OBSERVATION' in turn:
                    content = turn['OBSERVATION']
                    chat.append_message(chat.role_observation, content)
                if 'ASSISTANT' in turn:
                    content = turn['ASSISTANT']
                    chat.append_message(chat.role_assistant, content)

        return chat

    @classmethod
    def from_pack(
        cls,
        packs: Dict[str, List[str]],
        name: str,
        prompt_style: Optional[PromptStyle] = None,
    ) -> list:
        chat = cls(name=name, prompt_style=prompt_style)
        packs = cls._format_packs(packs)

        sys_pattern = re.compile(
            chat.system_template.format(r'(.*?)'), re.DOTALL)
        turn_pattern = re.compile(chat.turn_start.format(r'(\d+)'), re.DOTALL)
        human_pattern = re.compile(chat.role_template.format(
            chat.role_human).strip(), re.DOTALL)
        observe_pattern = re.compile(chat.role_template.format(
            chat.role_observation).strip(), re.DOTALL)
        assistant_pattern = re.compile(chat.role_template.format(
            chat.role_assistant).strip(), re.DOTALL)

        chats = []
        for input, output in zip(packs['input'], packs['output']):
            # system message
            sys_match = sys_pattern.search(input)
            if sys_match and sys_match.group(0):
                # system 指令只在首轮, 新增 chat 对象
                if len(chat.messages) > 0:
                    chats.append(chat)
                    chat = cls(name=name, prompt_style=prompt_style)

                input = input[sys_match.end():]
                chat.system_message = sys_match.group(1)

            # turn start
            turn_match = turn_pattern.search(input)
            if turn_match and turn_match.group(0):
                if name == 'chatglm2':
                    round_start = 1
                else:
                    round_start = 0

                if all(
                    [
                        len(turn_match.groups()) > 0,
                        int(turn_match.group(1)) == round_start,
                        len(chat.messages) > 0,
                    ]
                ):
                    chats.append(chat)
                    chat = cls(name=name, prompt_style=prompt_style)

                input = input[turn_match.end():]

            human_iter = human_pattern.finditer(input)
            observe_iter = observe_pattern.finditer(input)
            assistant_iter = assistant_pattern.finditer(input)
            human_match = next(human_iter, None)
            observe_match = next(observe_iter, None)
            assistant_match = next(assistant_iter, None)

            if not human_match and not observe_match:
                chat.append_message(chat.role_human, input)

            while human_match or observe_match:
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)
                input = cls._append_human_observation(
                    chat,
                    input,
                    human_match=human_match,
                    next_human_match=next_human_match,
                    observe_match=observe_match,
                    next_observe_match=next_observe_match,
                    assistant_match=assistant_match,
                )

                human_match = next_human_match
                observe_match = next_observe_match
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)

            if output:
                chat.append_message(chat.role_assistant, output)

        if chat.messages:
            chats.append(chat)

        return chats

    @classmethod
    def _append_human_observation(
        cls,
        chat,
        input: str,
        human_match: Optional[re.Match] = None,
        next_human_match: Optional[re.Match] = None,
        observe_match: Optional[re.Match] = None,
        next_observe_match: Optional[re.Match] = None,
        assistant_match: Optional[re.Match] = None,
    ) -> str:
        if observe_match:
            if observe_match.span()[0] > observe_match.span()[0]:
                human_str = input[observe_match.span()[1]: observe_match.span()[0]]
                observe_str = input[observe_match.span()[1]: assistant_match.span()[0]]
                chat.append_message(chat.role_human, human_str.strip())
                input_end = observe_match.span()[1]
                if observe_match.span()[0] < next_human_match.span()[0]:
                    chat.append_message(
                        chat.role_observation, observe_str.strip())
                    input_end = assistant_match.span()[1]
            else:
                human_str = input[observe_match.span()[1]: assistant_match.span()[0]]
                observe_str = input[observe_match.span()[1]: observe_match.span()[0]]
                chat.append_message(chat.role_observation, observe_str.strip())
                input_end = observe_match.span()[1]
                if observe_match.span()[0] < next_observe_match.span()[0]:
                    chat.append_message(chat.role_human, human_str.strip())
                    input_end = assistant_match.span()[1]
        else:
            if assistant_match:
                human_str = input[human_match.span(
                )[1]: assistant_match.span()[0]]
                input_end = assistant_match.span()[1]
            else:
                human_str = input[human_match.span()[1]:]
                input_end = len(input)
            chat.append_message(chat.role_human, human_str.strip())

        return input[input_end:]

    @classmethod
    def from_inout(
        cls,
        sample: Dict[str, str],
        name: str,
        prompt_style: Optional[PromptStyle] = None,
    ):
        chat = cls(name=name, prompt_style=prompt_style)
        input = sample['input']
        output = sample['output']

        sys_pattern = re.compile(
            chat.system_template.format(r'(.*?)'), re.DOTALL)
        turn_pattern = re.compile(chat.turn_start.format(r'(\d+)'), re.DOTALL)
        human_pattern = re.compile(chat.role_template.format(
            chat.role_human).strip(), re.DOTALL)
        observe_pattern = re.compile(chat.role_template.format(
            chat.role_observation).strip(), re.DOTALL)
        assistant_pattern = re.compile(chat.role_template.format(
            chat.role_assistant).strip(), re.DOTALL)

        input = turn_pattern.sub('', input)

        sys_match = sys_pattern.search(input)
        if sys_match and sys_match.group(0):
            input = input[sys_match.end():]
            chat.system_message = sys_match.group(1)

        human_iter = human_pattern.finditer(input)
        observe_iter = observe_pattern.finditer(input)
        assistant_iter = assistant_pattern.finditer(input)
        human_match = next(human_iter, None)
        observe_match = next(observe_iter, None)
        assistant_match = next(assistant_iter, None)
        next_human_match = next(human_iter, None)
        next_observe_match = next(observe_iter, None)

        while any(
            [
                human_match,
                observe_match,
                assistant_match,
            ]
        ):
            while any(
                [
                    human_match and human_match.span(
                    )[0] < assistant_match.span()[0],
                    observe_match and observe_match.span(
                    )[0] < assistant_match.span()[0],
                    next_human_match and next_human_match.span()[0] < assistant_match.span()[
                        0],
                    next_observe_match and next_observe_match.span()[0] < assistant_match.span()[
                        0],
                ]
            ):
                if not input:
                    break

                cls._append_human_observation(
                    chat,
                    input,
                    human_match=human_match,
                    next_human_match=next_human_match,
                    observe_match=observe_match,
                    next_observe_match=next_observe_match,
                    assistant_match=assistant_match,
                )

                human_match = next_human_match
                observe_match = next_observe_match
                next_human_match = next(human_iter, None)
                next_observe_match = next(observe_iter, None)

            if assistant_match and assistant_match.span():
                if observe_match:
                    if observe_match.span() and observe_match.span()[0] < human_match.span()[0]:
                        assistant_str = input[assistant_match.span()[1]: observe_match.span()[
                            0]]
                elif human_match:
                    if human_match.span():
                        assistant_str = input[assistant_match.span()[1]: human_match.span()[
                            0]]
                else:
                    assistant_str = input[assistant_match.span()[1]:]

                if assistant_str:
                    chat.append_message(chat.role_assistant, assistant_str)

            assistant_match = next(assistant_iter, None)

        if output:
            chat.append_message(chat.role_assistant, output)

        return chat

    def __hash__(self):
        return hash(self.id)

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        if not self.messages:
            self.messages = []

        if not self.name and not self.prompt_style:
            raise ValueError

        if not self.name and self.prompt_style == PromptStyle.RODIMUS_CHAT:
            logger.info(
                "The input parameter of the Chat object does not have `name`. By default, the `name` is RODIMUS_CHAT', format: \n"
                f'role_human: {self.role_human}\n'
                f'role_assistant: {self.role_assistant}\n'
                f'role_observation: {self.role_observation}\n'
                f'role_template: {self.role_template}\n'
                f'turn_start: {self.turn_start}\n'
                f'human_end: {self.human_end}\n'
                f'assistant_start: {self.assistant_start}\n'
                f'assistant_end: {self.assistant_end}\n'
                f'assistant_end_ids: {self.assistant_end_ids}\n'
                f'general_role_end: {self.general_role_end}\n'
                f'tool_template: {self.tool_template}\n'
                f'code_template: {self.code_template}\n'
                f'arithemetic_templte: {self.arithemetic_templte}\n'
                f'image_template: {self.image_template}\n'
                f'\n入参 `name` 支持: ``'
            )
            return

        if self.name in ['chatglm1', 'chatglm2'] or self.prompt_style == PromptStyle.CHATGLM:
            self.prompt_style = PromptStyle.CHATGLM
            self.role_template = '{}'
            self.role_human = '问：'
            self.role_assistant = '答：'
            self.turn_start = '[Round {}]\n'
            if self.name == 'chatglm1':
                self.general_role_end = '\n'
            else:
                self.general_role_end = '\n\n'

        elif self.name == 'chatglm3' or self.prompt_style == PromptStyle.CHATGLM3:
            self.prompt_style = PromptStyle.CHATGLM3
            self.system_template = '<|system|>\n {}'
            self.role_human = '<|user|>\n '
            self.role_assistant = '<|assistant|>\n '
            self.role_template = '{}'

        elif self.name == 'llama2' or self.prompt_style == PromptStyle.LLAMA2:
            self.prompt_style = PromptStyle.LLAMA2
            self.role_template = '{}'
            self.system_template = '[INST] <<SYS>>\n{}\n<</SYS>>\n\n'
            self.role_human = '[INST] '
            self.role_assistant = '[/INST] '
            self.human_end = ' '
            self.assistant_end = ' </s><s>'

        elif self.name == 'qwen':
            self.prompt_style = PromptStyle.CHATML
            self.role_template = '{}'
            self.system_template = '<|im_start|>system\n{}'
            if not self.system_message:
                self.system_message = 'You are a helpful assistant.'
            self.role_human = '<|im_start|>user\n'
            self.role_assistant = '<|im_start|>assistant\n'
            self.general_role_end = '<|im_end|>\n'

        elif self.name == 'baichuan':
            self.prompt_style = PromptStyle.BAICHUAN2
            self.role_template = '{}'
            self.system_template = '{}'
            self.role_human = '<token_id-195>'
            self.role_assistant = '<token_id-196>'

        if not self.system_template:
            self.system_template = '{}'

    def readable_messages(self) -> str:
        pass

    @property
    def prompt_str(self) -> str:
        return f'{self.prompt_inout["input"]}{self.prompt_inout["output"]}'

    @classmethod
    def _format_packs(cls, packs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        _packs = copy.deepcopy(packs)
        if len(_packs['input']) - 1 == len(_packs['output']):
            _packs['output'].append('')

        if len(_packs['input']) != len(_packs['output']):
            print(packs)
            raise ValueError(
                '输入 input 和 output 数量不匹配, '
                f'input num: {len(packs["input"])}, '
                f'output num: {len(packs["output"])}'
            )

        return _packs

    @property
    def prompt_inout(self) -> Dict[str, str]:
        packs = self._format_packs(self.prompt_pack)

        prompt_input = ''.join([f'{x}{y}' for x, y in zip(
            packs['input'][:-1], packs['output'][:-1])])
        prompt_input += packs['input'][-1]
        prompt_output = packs['output'][-1]

        return {
            'input': prompt_input,
            'output': prompt_output,
        }

    @property
    def prompt_pack(self) -> Dict[str, List[str]]:
        inputs = []
        outputs = []

        system_prompt = ''
        if self.system_message:
            system_prompt = self.system_template.format(self.system_message)

        if system_prompt:
            ret = system_prompt + self.general_role_end
        else:
            ret = ''

        if self.name in ['chatglm2']:
            round_start = 1
        else:
            round_start = 0

        for i, (role, message) in enumerate(self.messages):
            if self.name in ['chatglm1', 'chatglm2']:
                if i % 2 == 0:
                    ret += self.turn_start.format(i // 2 + round_start)

            role_end = self.general_role_end
            if role == self.role_assistant and self.assistant_end:
                role_end = self.assistant_end
            elif self.human_end:
                role_end = self.human_end

            ret += self.role_template.format(role) + message + role_end

            if role == self.role_assistant:
                if not message:
                    outputs.append('')
                else:
                    outputs.append(message + role_end)
                # input 需要连接 assistant role
                inputs[-1] += ret[: -len(message + role_end)]
            elif all(
                [
                    role == self.role_observation,
                    len(self.messages) > 1,
                    self.messages[i - 1][0] != self.role_assistant,
                ]
            ):
                continue
            else:
                inputs.append(ret)
            ret = ''

            if i == len(self.messages) - 1 and role != self.role_assistant:
                inputs[-1] += self.role_template.format(
                    self.role_assistant).strip()

        return {
            'input': inputs,
            'output': outputs,
        }

    @property
    def turns_num(self) -> int:
        return sum([1 if msg[0] == self.role_human else 0 for msg in self.messages])

    def to_json(self) -> dict:
        turns = []
        messages = []
        turn = {}
        for msg in self.messages:
            if msg[0] == self.role_assistant:
                messages.append({'role': 'ASSISTANT', 'content': msg[1]})
                turn['ASSISTANT'] = msg[1]
                turns.append(turn)
                turn = {}

            if msg[0] == self.role_human:
                messages.append({'role': 'HUMAN', 'content': msg[1]})
                turn['HUMAN'] = msg[1]

            if msg[0] == self.role_observation:
                messages.append({'role': 'OBSERVATION', 'content': msg[1]})
                turn['OBSERVATION'] = msg[1]

        if self.messages[-1][0] == self.role_human:
            messages.append({'role': 'ASSISTANT', 'content': ''})
            turn['ASSISTANT'] = ''
            turns.append(turn)

        result = self.origin_json or {}
        result.update(
            {
                'id': self.id,
                'name': self.name,
                'source': self.source,
                'lang': self.lang,
                'topic': self.topic,
                'system_template': self.system_template,
                'system_message': self.system_message,
                'turns': turns,
                'messages': messages,
            }
        )

        return result

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        if not message:
            message = ''
        self.messages.append([role, message])

    def to_openai_api_messages(self) -> List[dict]:
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return copy.deepcopy(self)
