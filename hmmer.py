import urllib.parse as p
from urllib.request import Request, urlopen
import json

# create a job to search domain
def sequence_search(seq):
    url = 'http://www.cathdb.info/search/by_funfhmmer'
    values = {'fasta': seq}
    headers = {'Accept': 'application/json'}

    data = p.urlencode(values)
    data = data.encode('ascii')
    req = Request(url, data, headers)

    with urlopen(req) as response:
        res = json.loads(response.read())

    return res["task_id"]

# check if the job is done
def check(task_id):
    url = 'http://www.cathdb.info/search/by_funfhmmer/check/{}'.format(task_id)
    headers = {'Accept': 'application/json'}

    req = Request(url, None, headers)

    with urlopen(req) as response:
        res = json.loads(response.read())

    return res["data"]["status"] == "done" and res["success"] == 1

# retrieve results
def retrieve_result(task_id):
    url = 'http://www.cathdb.info/search/by_funfhmmer/results/{}'.format(task_id)
    headers = {'Accept': 'application/json'}

    req = Request(url, None, headers)

    with urlopen(req) as response:
        res = response.read()

        if res:
            res = json.loads(res)
        else:
            return

    return res