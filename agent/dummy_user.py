from typing import Optional

from common.common_utils import colorful, read_json, save_to_json
from tool.openai_api import chatgpt

INSTRUCTION = """
This is a clarification scenario. Given the SPARQL query, when someone asks you to clarify your intention, please give your response.

Here is a example:

SPARQL: SELECT DISTINCT ?x WHERE { FILTER (?x != m.0bv70) ?e0 type.object.name "Alice Walker"@en . ?e0 people.person.profession ?x . }
Entity Description: {"Alice Walker": "Alice Malsenior Walker is an American author and activist. She wrote the critically acclaimed novel The Color Purple for which she won the National Book Award and the Pulitzer Prize for Fiction. She also wrote Meridian and The Third Life of Grange Copeland among other works."}

Input question: Which Alice Walker do you want, the British fencer, the American author, or the film character?
Your response: I want the American author and activist.

Guideline:
1. MUST NOT reveal any information about SPARQL, provide response based on the input question.
2. MUST use the exact entity name from the entity description's key field (e.g. use "Alice Walker" from the key field rather than "Alice Malsenior Walker" from the description).
"""


class DummyUser:
    """
    Given the golden sparql, generate a response for the clarification question.
    sparql should contain the xxx.
    """

    def __init__(self, qid, sparql, entity_desc, model_name):
        self.qid = qid
        self.sparql = sparql.replace("ns:", "").replace("\n", " ")
        self.entity_desc = str(entity_desc).replace("\\n", " ")
        self.model_name = model_name
        """
        {
            "role": "user",
            "content": "..." # the input question
        },
        {
            "role": "assistant",
            "content": "..." # my response
        }
        """
        self.message = [{"role": "system", "content": INSTRUCTION}]

    def __call__(self, question):
        if len(self.message) == 1:
            content = "This is the real case, please respond based on the SPARQL and entity description.\n\n"
            content += f"SPARQL: {self.sparql}\nEntity Description: {self.entity_desc}\n\n"
            content += f"Input question: {question}"
        else:
            content = f"Input question: {question}"

        self.message.append({"role": "user", "content": content.strip()})

        response = chatgpt(
            messages=self.message,
            model=self.model_name,
        )
        response = response["choices"][0]["message"]["content"].strip()

        # debug
        print(colorful("DummyUser: ", color="red"), end="")
        print(response.replace("\n", "\\n"))

        self.message.append({"role": "assistant", "content": response})
        return response

    def to_dict(self):
        return {
            "qid": self.qid,
            "sparql": self.sparql,
            "entity_desc": self.entity_desc,
            "model_name": self.model_name,
            "messages": self.message,
        }

    @classmethod
    def load_from_file(cls, path):
        d = read_json(path)
        c = cls(d["qid"], d["sparql"], d["entity_desc"], d["model_name"])
        c.message = d["messages"]
        return c

    def gen_clear_q(self) -> Optional[str]:
        sparql = self.sparql.replace("PREFIX  <http://rdf.freebase.com/ns/>", "").strip()
        history = []
        for message in self.message:
            if message["role"] == "user":
                inp_from_qa = message["content"].split("Input question: ")[1]
                history.append(f"Clarification question from QA system: {inp_from_qa}")
            elif message["role"] == "assistant":
                history.append(f"Response from user: {message['content']}")
        history = "\n".join(history)

        if history == "":
            return None

        prompt = f"""
This is a question generation task. Based on the user's intention and clarification history, please generate a clear and concise question.

Guidelines:
1. The question should accurately reflect the user's information need
2. Do not include any technical terms or database-specific language
3. Make the question natural, specific and brief - as if asked by a real user
4. The question should incorporate the clarifications from the previous interaction

SPARQL: {sparql}

Interaction history:
{history}

Now, please generate the clear question about the user's intention.
        """.strip()

        response = chatgpt(
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o",
            temperature=1,
            n=1,
        )
        res_list = [r["message"]["content"].strip() for r in response["choices"]]
        return res_list


def test_interaction():
    p = "dataset_processed-v2.0/webqsp/test-300/chain_len_1.json"
    data = read_json(p)
    for d in data:
        if d["id"] in ["WebQTest-2030.P0"]:
            user = DummyUser(d["id"], d["sparql"], d["entity_desc"], "gpt-4o")
            r = user("which area 51 do you want to know, the one in New Mexico or the one in Nevada?")
            print(r)
            r = user("which one do you want to know, the town name or the coordinates?")
            print(r)


def test_gen_clear_q():
    p = "save/debug/webqsp/gpt-4o-2024-11-20/WebQTest-99.P0-dummy_user.json"
    user = DummyUser.load_from_file(p)
    clear_q = user.gen_clear_q()
    new_dict = user.to_dict()
    new_dict["clear_q"] = clear_q
    save_to_json(new_dict, p)


if __name__ == "__main__":
    # python dummy_user.py
    # test_interaction()
    test_gen_clear_q()
