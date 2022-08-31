import os
from flask import jsonify
from google.cloud import storage, firestore
import firebase_admin
from firebase_admin import db
from datetime import datetime
import maya
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

import uuid
import gc
from collections import Counter

firebase_admin.initialize_app(options={
    'databaseURL': 'https://snaplatform.europe-west1.firebasedatabase.app',
})

class SNAP:

    def __init__(self):
        self._df = None
        self._ios_data = None
        self.nodes_list = []
        self.edges_list = []
        self.degree_centrality = {}
        self.closeness_centrality = {}
        self.betweenness_centrality = {}
        self.clustering = {}
        self.shortest_path = {}
        self.global_measures = {}
        self.local_measures = {}
        self.statistics = {}
        self.community = {}

    def read_txt(self, path, startDate, endDate, fileDateFormat):
        '''Iterates trough all messages inside .txt file and creates messages
        instances that are loaded inside self.messages.'''

        messages = []

        # iPhone
        # [14/06/18, 12:47:32] sender: content
        # [14/06/18, 3:25:32 PM] sender: content

        # Android
        # 19/02/18, 10:27 PM - sender: content
        # 05/11/17, 09:33 - sender: content

        storage_client = storage.Client()
        bucket = storage_client.get_bucket("snaplatform.appspot.com")
        blob = bucket.get_blob(path)

        f = blob.download_as_text(encoding="utf-8")

        data = f.splitlines()
        self._ios_data = data[0][0] == '['

        for line in data:
            if not line.isspace():
                try:
                    message = self._construct_message(line, fileDateFormat)
                    if startDate.replace(tzinfo=None) <= message["Date"].replace(tzinfo=None) <= endDate.replace(tzinfo=None):
                        messageDate = str(maya.MayaDT.from_datetime(message["Date"]))
                        message["Date"] = messageDate[:len(messageDate) - 4]
                        messages.append(message)
                    
                except ValueError as NotFoundDatetime:
                    # Add content to the last message
                    if len(messages) > 0:
                        messages[-1]['Content'] += f'\n{line.strip()}'
        self._df = pd.DataFrame(messages) 
        
    def _construct_message(self, line, fileDateFormat):
        '''Removes data from each line inside the file and returns a Message'''
        datetime = self._get_datetime_from_line(line, fileDateFormat)
        sender = self._get_sender_from_line(line)
        content = self._get_content_from_line(line)
        return {'Date': datetime, 'Sender': sender, 'Content': content}

    def _get_datetime_from_line(self, line, fileDateFormat):
        '''Extracts datetime data from a line'''
        datetime = None

        if self._ios_data == True:
            endIndex = line.find(']')
            if endIndex == -1:
                raise ValueError(
                    'Could not parse datetime string due continual message')

            datetime = line[1:endIndex]
            if datetime[0] == '[':
                datetime = datetime[1:]
        else:
            data = line.strip().split(" - ", 1)
            if len(data) > 1:
                datetime = data[0]
            else:
                raise ValueError(
                    'Could not parse datetime string continual message')

        try:
            if fileDateFormat == "DMY":
                datetime = maya.parse(datetime, day_first=True, year_first=False).datetime()
            elif fileDateFormat == "MDY":
                datetime = maya.parse(datetime, day_first=False, year_first=False).datetime()
            else:   
                datetime = maya.parse(datetime, day_first=False, year_first=True).datetime()

        except ValueError:
            raise ValueError('Could not parse datetime string')
            
        return datetime

    def _get_sender_from_line(self, line):
        '''Extracts sender data from a line'''
        sender = None

        if self._ios_data == True:
            endDatetime_index = line.find(']')
            startSender = line[endDatetime_index+2:]
            endSender_index = startSender.find(':')
            if endSender_index == -1:
                return np.nan

            sender = startSender[:endSender_index]

        else:
            data = line.strip().split(" - ", 1)[1].split(":", 1)
            if len(data) > 1:
                sender = data[0]
            else:
                return np.nan

        return sender

    def _get_content_from_line(self, line):
        '''Extracts content data from a line'''
        content = None

        if self._ios_data == True:
            endDatetime_index = line.find(']')
            startSender = line[endDatetime_index+2:]
            endSender_index = startSender.find(':')

            content = startSender[endSender_index+1:]
        else:
            data = line.strip().split(" - ", 1)[1].split(":", 1)
            if len(data) > 1:
                content = data[1]
            else:
                content = data[0]

        return content.strip()

    def clean(self):
        # need to remove media content info ("removed")
        if self._df is not None:
            df = self._df.copy()
            df = self._df.dropna()

            for i, row in df.iterrows():
                df.at[i, 'Sender'] = str(row['Sender'])

            return df

        return None   

    def to_csv(self, path):
        if self._df is not None:
            df = self.clean()
            df.to_csv('gs://snaplatform.appspot.com/' + path + '@processed.csv',
                            encoding='utf-8-sig', sep=',', index=False)

    def to_network(self):
        if self._df is not None:
            df = self.clean()

            self.statistics["topActivity"] = [{'node': item[0], 'messages': item[1]} for item in df["Sender"].value_counts().head(10).to_dict().items()]

            # Create new dataframe
            newDf = pd.DataFrame(columns=['From', 'To'])

            newDf['From'] = df['Sender']
            newDf['To'] = df['Sender'].shift()

            # Drop null values from the shift
            newDf = newDf.dropna()

            nodes = df['Sender'].unique()
            whatsappedges_list = [tuple(row) for row in newDf.to_numpy()]

            self.nodes_list = nodes
            self.edges_list = whatsappedges_list

    def DegreeCentrality(self):
        # Compute the degree centrality for nodes.
        # explanation: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html
        # source: https://www.geeksforgeeks.org/network-centrality-measures-in-a-graph-using-networkx-python/

        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        self.degree_centrality = nx.degree_centrality(whatsapp_graph)

    def ClosenessCentrality(self):
        # Compute closeness centrality for nodes.
        # explanation: https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
        # source: https://www.geeksforgeeks.org/network-centrality-measures-in-a-graph-using-networkx-python/

        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        self.closeness_centrality = nx.closeness_centrality(whatsapp_graph)
        
    def BetweennessCentrality(self):
        # Compute closeness centrality for nodes.
        # explanation: https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
        # source: https://www.geeksforgeeks.org/network-centrality-measures-in-a-graph-using-networkx-python/

        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        self.betweenness_centrality = nx.betweenness_centrality(
            whatsapp_graph, normalized=True, endpoints=False)

    def ShortestPath(self):
        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        edges_list = list(set(self.edges_list))

        whatsapp_graph.add_edges_from(edges_list)

        self.shortest_path = nx.shortest_path(whatsapp_graph)       

    def FindGlobalMeasures(self):
        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        density = nx.density(whatsapp_graph)
        numberOfSelfLoops = nx.number_of_selfloops(whatsapp_graph)

        try:
            diameter = nx.diameter(whatsapp_graph)
            radius = nx.radius(whatsapp_graph) 
            center = list(nx.center(whatsapp_graph))
        except nx.NetworkXException:
            whatsapp_graph = whatsapp_graph.to_undirected()
            diameter = nx.diameter(whatsapp_graph)
            radius = nx.radius(whatsapp_graph) 
            center = list(nx.center(whatsapp_graph))
        self.global_measures = { 'radius':radius, 'diameter':diameter, 'density':density,'center':center, 'numberOfSelfLoops': numberOfSelfLoops }


    def FindLocalMeasures(self):
        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        transitivity = nx.transitivity(whatsapp_graph)

        reciprocity = nx.reciprocity(whatsapp_graph)

        self.clustering = nx.clustering(whatsapp_graph)

        average_clustering = nx.average_clustering(whatsapp_graph)
        
        self.local_measures = { 'transitivity':transitivity, 'reciprocity':reciprocity, 'average_clustering':average_clustering }

    def FindCommunities(self):
        whatsapp_graph = nx.DiGraph(name='whatsapp')

        for node in self.nodes_list:
            whatsapp_graph.add_node(node)

        whatsapp_graph.add_edges_from(self.edges_list)

        communities = list(greedy_modularity_communities(whatsapp_graph))

        group = 0
        for community in communities:
            for node in community:
                self.community[node] = str(group)
            group += 1
        

def start(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
   # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

   # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Origin': '*'
    }

    try:
        request_json = request.get_json()
        if request_json and 'conversation' in request_json:
            conversation = request_json['conversation']
            projectId = conversation['projectId']
            conversationId = conversation['conversationId']

            conversationFile = conversation['conversationFile']
            fileDateFormat = conversationFile.get('fileDateFormat', "") 
            filePath = conversationFile.get('filePath', "")
            fileName = conversationFile.get('fileName', "")
            fileNameWithoutExtension, extension = os.path.splitext(fileName)
            storageSourcePath = filePath+fileName
            storageProcessedPath = filePath + fileNameWithoutExtension

            if conversationFile['isFromSources'] == True:
                storageProcessedPath = 'Conversations/' + conversationId + '/' + fileNameWithoutExtension 
                storageSourcePath = conversationFile['storageSourcePath']

            minDate = conversation['minDate']
            maxDate = conversation['maxDate']
            startDate = datetime.fromtimestamp(int(minDate) / 1000.0)
            endDate = datetime.fromtimestamp(int(maxDate) / 1000.0)

            # Process file
            sna = SNAP()
            sna.read_txt(path=storageSourcePath,startDate=startDate,endDate=endDate,fileDateFormat=fileDateFormat)
            sna.to_csv(path=storageProcessedPath)
            sna.to_network()

            # Calculate
            sna.DegreeCentrality()
            sna.ClosenessCentrality()
            sna.BetweennessCentrality()
            sna.ShortestPath()
            sna.FindGlobalMeasures()
            sna.FindLocalMeasures()
            sna.FindCommunities()

            # Save to database

            nodes_list = []
            edges_list = []

            # Fix for firestore
            if str(type(sna.nodes_list)) == "<class 'numpy.ndarray'>":
                nodes_list = (sna.nodes_list).tolist()
            else:
                nodes_list = sna.nodes_list

            if str(type(sna.edges_list)) == "<class 'numpy.ndarray'>":
                edges_list = (sna.edges_list).tolist()
            else:
                edges_list = sna.edges_list

            nodes_list = [{'id': str(uuid.uuid1()), 'label': item, 'centrality': {'degree': sna.degree_centrality.get(item, 0), 'closeness': sna.closeness_centrality.get(item, 0), 'betweenness': sna.betweenness_centrality.get(item, 0)
            }, 'clustering': sna.clustering.get(item, 0), 'group': sna.community.get(item, 0)} for item in nodes_list]

            countMessages = dict(Counter(e for e in edges_list)) 
            edges_list = list(set(edges_list))
            edges_list = [{'from': item[0], 'to': item[1], 'weight': countMessages.get(item, 0)} for item in edges_list]

            # Create source
            source_data = {
                'id': conversationId,
                'name': fileNameWithoutExtension,
                'owner': conversation['fileOwner'],
                'fileDateFormat': fileDateFormat,
                'uploadDate': maya.now().datetime(),
                'uploadedBy': conversation['creator'],
                'storageSourcePath': filePath+fileName,
                'storageProcessedPath': storageProcessedPath + '@processed.csv',
                }

            if conversationFile['isFromSources'] == True:
                source_data = {
                    'id': conversationFile['id'],
                    'name': conversationFile['name'],
                    'owner': conversationFile['owner'],
                    'fileDateFormat': fileDateFormat,
                    'uploadDate': conversationFile['uploadDate'],
                    'uploadedBy': conversationFile['uploadedBy'],
                    'storageSourcePath': conversationFile['storageSourcePath'],
                    'storageProcessedPath': storageProcessedPath + '@processed.csv',
                    'isFromSources': True
                }

            # Create conversation
            conversation_ref = firestore.Client().collection("Conversations").document(conversationId)

            source_ref = None
            isConversationExist = False
            if conversation['futureUse'] == True:
                source_ref = firestore.Client().collection("Sources").document(conversationId)
                doc = source_ref.get()
                if doc.exists:
                    isConversationExist = True
                else:
                    source_ref.set({'conversation': conversation_ref, 'source': source_data})
 
            conversation_ref.create({
                'title': conversation['title'],
                'description': conversation['description'],
                'creator': conversation['creator'],
                'createdAt': maya.now().datetime(),
                'isPublished': False,
                'nodes': nodes_list,
                'edges': edges_list, 
                'globalMeasures': sna.global_measures,
                'localMeasures': sna.local_measures,
                'statistics': sna.statistics,
                'source': source_data,
            })
            conversation_ref_rt = db.reference('/Conversations/'+conversationId)

            for key, value in sna.shortest_path.items():
                conversation_ref_rt.child('shortestPath').update({key:value})
            

            # conversation_ref_rt.set({'shortestPath': sna.shortest_path})
        
            # Set project with conversation ref
            project_ref = firestore.Client().collection('Projects').document(projectId)
            doc = project_ref.get()
            if doc.exists:
                if source_ref is not None:
                    project_ref.update({'conversations': firestore.ArrayUnion([conversation_ref]), 'sources': firestore.ArrayUnion([source_ref])})
                else:
                    project_ref.update({'conversations': firestore.ArrayUnion([conversation_ref])})
            else:
                project_ref.set({'conversations': firestore.ArrayUnion([conversation_ref]), 'sources': firestore.ArrayUnion([source_ref])})

            gc.collect()

            return (jsonify({'status': True , 'message': 'Analyzed conversation successfully'}), 200, headers)

    except Exception as e:
        print("Exception!")
        print(e)

    return (jsonify({'status': False , 'message': 'Failed to analyze conversation'}), 400, headers)
