from traceback import print_exc

import requests

from common.constant import API_SERVER_FB

requests.packages.urllib3.disable_warnings()


def get_actions_api(db):
    if db == "fb":
        _base = API_SERVER_FB
        n_results = 10
    else:
        raise ValueError(f"db: {db} not supported.")

    base_url = f"{_base}/{db}/"

    def test():
        url = base_url + "test"
        response = requests.get(url, timeout=5, verify=False)
        assert response.status_code == 200, f"Connection to {db} failed."

    test()

    def SearchNodes(query="", n_results=n_results, str_mode=True):
        url = base_url + "SearchNodes"
        data = {
            "query": query,
            "n_results": n_results,
            "str_mode": str_mode,
        }
        response = requests.post(url, data=data, timeout=120, verify=False)
        return response.json()

    def SearchGraphPatterns(
        sparql="", semantic="", topN_return=10, return_fact_triple=True, force_update=False
    ):
        """
        def SearchGraphPatterns(
            sparql: str = None,
            semantic: str = None,
            topN_vec=400,
            topN_return=10,
            force_update=False,
        ):
        """
        url = base_url + "SearchGraphPatterns"
        data = {
            "sparql": sparql,
            "semantic": semantic,
            "return_fact_triple": return_fact_triple,
            "topN_return": topN_return,
            "force_update": force_update,
        }
        try:
            response = requests.post(url, data=data, timeout=600, verify=False)
            return response.json()
        except Exception as e:
            print_exc()
            print("data,", data)
            return None

    def ExecuteSPARQL(sparql="", str_mode=True):
        url = base_url + "ExecuteSPARQL"
        data = {
            "sparql": sparql,
            "str_mode": str_mode,
        }
        response = requests.post(url, data=data, timeout=600, verify=False)
        return response.json()

    return SearchNodes, SearchGraphPatterns, ExecuteSPARQL


def test_fb():
    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api("fb")

    print(SearchNodes("Alice Walker", str_mode=False))

    s = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e0 ns:type.object.name "Atlanta Falcons"@en . ?e0 ns:sports.sports_team.roster ?c . ?c ns:sports.sports_team_roster.position ?e1 . ?e1 ns:type.object.name "Quarterback"@en . ?c ns:sports.sports_team_roster.number ?e . }',
        semantic="passing attemptaas",
    )
    print(s)

    s = ExecuteSPARQL('SELECT ?e WHERE { ?e ns:type.object.name "Barack Obama"@en . }')
    print(s)


def test_kqapro():
    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api("kqapro")

    print(SearchNodes("electronic dance music"))

    # data, {'sparql': 'SELECT ?e WHERE {}', 'semantic': 'ISNI', 'return_fact_triple': True, 'topN_return': 10, 'force_update': False}
    r = SearchGraphPatterns("SELECT ?e WHERE {}", "ISNI")
    print(r)


if __name__ == "__main__":
    # python api/api_db_client.py
    test_fb()
    # test_kqapro()
