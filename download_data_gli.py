import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken="YOUR_AUTH_TOKEN_HERE")

dl_list_file_entities = syn.get_download_list()