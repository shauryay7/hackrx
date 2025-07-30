from app.query_answering import search_document

query = "Does this policy cover maternity benefits?"
response = search_document(query)

from pprint import pprint
pprint(response)
