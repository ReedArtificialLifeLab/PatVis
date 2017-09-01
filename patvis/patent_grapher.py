""" TODO:
  - change string variables into actual types (i.e. python classes)
  - Abstract out traits, see traits subpackage for an example, specifically __init__.py & gpe.py
"""

import sys, csv, os
import networkx as nx
from pygraphviz import AGraph
from pymongo import MongoClient
from collections import defaultdict
from getpass import getpass
import matplotlib.pyplot as plt
from patvis import get_family_and_friends
import patvis.patent_grapher_traits as pg_traits
#from sklearn.decomposition import PCA
# import tty, termios
# import sys
# import _thread
# import time

# # cross-platform async getkeypress from RC
# try:
#     from msvcrt import getch  # try to import Windows version
# except ImportError:
#     def getch():   # define non-Windows version
#         fd = sys.stdin.fileno()
#         old_settings = termios.tcgetattr(fd)
#         try:
#             tty.setraw(sys.stdin.fileno())
#             ch = sys.stdin.read(1)
#         finally:
#             termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#         return ch

# def keypress():
#     global char
#     char = getch()
    
# def has_user_aborted(char):
#     if char is not None:
#         if char == 'q':
#             char = None
#             return True
#         else:
#             _thread.start_new_thread(keypress, ())
#             char = None
#             return False

faf = get_family_and_friends()

class PatentGrapher(object):
    """
    Class for streamlining patent genealogy visualization
    Sensible defaults for graphing patents from our database are stored in 
    class attributes while patent specific data is scoped within the main
    'build_and_draw_pno_graph' function
    Class attributes are exposed to the user as graphing environment variables
    """

    def __init__(self):
        # hardcoded for now
        # this class is only designed to recover info from a specific database on a specific computer
        self.set_database('markov.reed.edu',27017,"patents")
        self.set_trait_type("category")
        self.start_year=1976
        self.end_year=2015
        self.color_edges_by_parent = False
        self.include_ancestors = False
        self.include_descendants = True
        self.draw_each_gen = False
        self.grayscale = False
        self.show_parameters = True
        self.edge_width = 5
        self.outfile_path = os.getcwd()
        self.node_width = 10
        self.node_height = 10
        self.size = (7.5,10)
        self.layout_prog = 'dot'
        self.prune_to_target = False
        self.pruning_target = 1000

    def __str__(self):
        return ("\nCurrent Graphing Environment:\n\t" +
                '\n\t'.join("{0}: {1}".format(variable, value) for variable,value in sorted(self.__dict__.items()) if variable not in [ "criteria", "projection", "pno_field", "trait" ]) +
                "\n")

    def set_database(self, address, port, database):
        u = input('Please Enter Your Username:')
        p = getpass('Please Enter Your Password:')
        if address == None or port == None:
            self.conn = MongoClient(username=u, password=p)
        else:
            self.conn = MongoClient(address,port,username=u, password=p)
        self.db = self.conn[database]

    def set_trait_type(self,trait_type):
        """sets patent grapher object trait type to be used in future graphs"""

        if trait_type not in pg_traits.traits:
            raise Exception("Invalid Trait Type")
        else:
            self.trait = pg_traits._get_trait(trait_type)

    def build_and_draw_pno_graph(self, originalpno, ngenerations, ofile, threshold=0):
        """
        *main function*
        builds network originating from 'originalpatn' with specified number of generations, 'gen',
        and patent inclusion threshold, 'threshold', and then writes the dot-generated image to 'ofile'
        """

        # add qualifications to filename based on graph parameters
        ofile = self.generate_filename(ofile)

        # build graph from bfs through database
        # if draw_each_gen is on, this will call draw_graph on each iteration
        aborted, G, color_scheme = self.build_graph(ngenerations, originalpno, ofile, threshold)

        # If user typed 'q' to abort then don't bother trying to draw G
        # if aborted:
        #     return

        if self.prune_to_target:
            G, threshold = self.prune_graph(G, originalpno, threshold)

        self.draw_graph(G, color_scheme, ofile, ngenerations, threshold, originalpno)

    def build_graph(self, ngenerations, originalpno, ofile, threshold):

        # init graph, color scheme and db query templates
        db = self.db
        G = nx.DiGraph()
        color_scheme = None # dict mapping trait values to colors

        # # set async keypress listener 
        # global char
        # char = None
        # _thread.start_new_thread(keypress, ())
        
        # add first patent
        search = dict(self.trait.criteria)
        search[self.trait.pno_field] = originalpno
        patn = db[self.trait.collection].find_one(search,self.trait.projection)
        if patn == None:
            print("Original Patent No. {} missing Necessary Fields".format(originalpno))
            return

        next_gen_ancestors = { originalpno : map(int, patn[self.trait.cited]) }
        next_gen_descendants = { originalpno : map(int, patn[self.trait.citedby]) }

        G.add_node( originalpno, self.get_node_attributes(patn, {'shape' : 'star', 'width' : self.node_width, 'height' : self.node_height, 'ncites' : len(patn[self.trait.citedby]) }) )

        # bfs through the graph of all patents, starting from originalpno
        for generation in range(ngenerations):

            potential_ancestors = next_gen_ancestors
            potential_descendants = next_gen_descendants
            next_gen_ancestors = {}
            next_gen_descendants = {}
            undersizedpatns,badpatns,ancsprocessed,descsprocessed = 0,0,0,0

            # add ancestors
            if self.include_ancestors:
                # for each edge from last gen to this gen
                for descpno,ancpnos in potential_ancestors.items():
                    for ancpno in ancpnos:

                        ancsprocessed += 1
                        
                        search = dict(self.trait.criteria)
                        search[self.trait.pno_field] = ancpno
                        patns = db[self.trait.collection].find(search,self.trait.projection)

                        assert(patns.count() <= 1)
                        if patns.count() == 1:
                            anc = patns.next()
                        
                            if len(anc[self.trait.citedby]) >= threshold:

                                if not G.has_node(ancpno):
                                    G.add_node(ancpno, self.get_node_attributes(anc, { 'ncites' : len(anc[self.trait.citedby]) }))
                                    
                                G.add_edge(ancpno, descpno, penwidth=self.edge_width, arrowhead="none")
                                                                
                                next_gen_ancestors[ancpno] = map(int, anc[self.trait.cited])
                                                                
                            else:
                                undersizedpatns += 1
                        else:
                            badpatns += 1

                        # if has_user_aborted(char):
                        #     return True, None, None, None, None

            # add descendants
            if self.include_descendants:
                # for each edge from last gen to this gen
                for ancpno,descpnos in potential_descendants.items():
                    for descpno in descpnos:

                        descsprocessed += 1
                        
                        search = dict(self.trait.criteria)
                        search[self.trait.pno_field] = descpno
                        desc = db[self.trait.collection].find_one(search,self.trait.projection)

                        if desc != None:
                            
                            if len(desc[self.trait.citedby]) >= threshold:

                                if not G.has_node(descpno):
                                    G.add_node(descpno, self.get_node_attributes(desc,{ 'ncites' : len(desc[self.trait.citedby]) }))

                                G.add_edge(ancpno, descpno, penwidth=self.edge_width, arrowhead="none")
                                
                                next_gen_descendants[descpno] = map(int, desc[self.trait.citedby])
                                                                
                            else:
                                undersizedpatns += 1
                        else:
                            badpatns += 1

                        # if has_user_aborted(char):
                        #     return True, None, None, None, None

            # print stats for generation
            print("Finished Generation {}.".format(generation+1))
            print("Patents Under Threshold: {}".format(undersizedpatns))
            print("Patents Missing Fields: {}".format(badpatns))
            print("Ancestors Processed this Generation: {}".format(ancsprocessed))
            print("Descendants Processed this Generation: {}".format(descsprocessed))
            print("Nodes in Graph: {}".format(G.order()))

            if self.draw_each_gen:
                color_scheme = self.draw_graph(G, color_scheme, ofile, generation, threshold)
                
        return False, G, color_scheme

    def draw_graph(self, G, color_scheme, ofile, generation, threshold, originalpno):
        """Writes graph to file with some aesthetics and graph descriptions"""

        print("Coloring...")
        color_scheme = self.trait.generate_color_scheme(G, color_scheme)
        G = self.color_graph(G, color_scheme)
        
        # add annotations, graph aesthetic attributes and write to file
        print("Laying out graph and drawing...")
        plt.figure(figsize=self.size)
        pos = self.position_nodes(G)

        # the graph is colored according to dot format, we have to convert to input to matplotlib
        node_colors = [ attr['fillcolor'] for _,attr in G.nodes_iter(data=True) ]
        edge_colors = [ attr['color'] for _,_,attr in G.edges_iter(data=True) ]
        node_sizes = [ 25 if node != originalpno else 300 for node in G.nodes_iter() ] # 25 is sensible default size for networkx drawing

        # split the original filename of the format <name>.<extension>
        filename, file_extension = os.path.splitext(ofile)
        name = filename.split('_')[0]

        # parameter for drawing parameters onto pdf output
        if self.show_parameters:
            info = "Patent: " + filename + "\nGenerations: " + str(generation)
            if self.color_edges_by_parent:
                info += "\nColor Edges By: parent"
            else:
                info += "\nColor Edges By: child"
            info += "\nThreshold: " + str(threshold) + "\nNodes: " + str(G.order())
            plt.annotate(info, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

        nx.draw(G,
                pos=pos,
                node_color=node_colors,
                edge_color=edge_colors,
                node_size=node_sizes,
                arrows=False)
        
        # create a directory if it doesn't already exist
        # and place the saved pdf there
        dir = str(self.outfile_path) + str('/') + str(filename)
        if not os.path.exists(dir):
            os.mkdir(dir)
        fn = dir + str('/') + str(filename) + str(generation+1)
        plt.savefig(fn + str(file_extension))

        # write a dot file to save the node and edge info for reuse
        nx.nx_agraph.write_dot(G, fn + '.dot')
        return color_scheme

    def color_graph(self, G, color_scheme):
        """
        colors the graph 'G' with specified color scheme
        """
        node_colors,edge_colors = [],[]
        for node in G.nodes_iter():
            maintrait = G.node[ node ][self.trait.trait_type]
            if maintrait in color_scheme:
                G.node[ node ]['fillcolor'] = color_scheme[maintrait]
                if self.color_edges_by_parent:
                    for o,i in G.out_edges_iter([node]):
                        G.edge[o][i]['color'] = color_scheme[maintrait]
                else:
                    for o,i in G.in_edges_iter([node]):
                        G.edge[o][i]['color'] = color_scheme[maintrait]
            else:
                G.node[ node ]['fillcolor'] = '#000000'
                if self.color_edges_by_parent:
                    for o,i in G.out_edges_iter([node]):
                        G.edge[o][i]['color'] = '#000000'
                else:
                    for o,i in G.in_edges_iter([node]):
                        G.edge[o][i]['color'] = '#000000'
                
        return G, node_colors, edge_colors

    def position_dot(self, G):
        """
        Converts to pygraphviz format and Induces levels for dot format
        We only use AGraph because networkx graphs do not support induced subgraph levels
        """

        # group nodes by years
        year_subgraph_arrays = defaultdict(list)
        years = []
        for node,data in G.nodes_iter(data=True):
            #year_subgraph_arrays[ data['year'] ].append(data['year'])
            year_subgraph_arrays[ data['year'] ].append(node)
            #if not G.has_node(data['year']):
            #    years.append(data['year'])

        A = nx.nx_agraph.to_agraph(G)

        # induce subgraphs with rank=same
        # this is what forces patents in the same year to appear on the same level
        for subgraph in year_subgraph_arrays.values():
            A.add_subgraph(subgraph,rank="same")

        A.graph_attr.update(ratio="fill",fontsize=200,ranksep="equally")
        A.layout(self.layout_prog)
            
        return { int(node) : tuple(map(int, node.attr['pos'].split(','))) for node in A.nodes_iter() }

    def get_node_attributes(self, patent, extras={}):
        """node attributes for different trait types"""

        attrib = self.trait.get_node_attributes(patent)
        for k,v in list(extras.items()):
            attrib[k] = v
        return attrib

    def prune_graph(self, G, originalpno, threshold):
        """
        Increases threshold until graph size is below target size
        """
        current_threshold = threshold
        print("Pruning...")
        while G.order() > self.pruning_target:
            current_threshold += 1
            print(threshold,G.order())
            G.remove_nodes_from( [ node for node,attr in G.nodes_iter(data=True) if (attr['ncites'] < current_threshold and node != originalpno) ] )
            for nodes in nx.connected_components(G.to_undirected()):
                if originalpno not in nodes:
                    G.remove_nodes_from(nodes)
        print("Final Size: ", G.order())
        return G, current_threshold

    def position_nodes(self, G):
        """calls the appropriate layout function"""
        
        if self.layout_prog in [ 'twopi', 'sfdp', 'fdp', 'neato', 'circo' ]:    
            return nx.drawing.nx_agraph.graphviz_layout(G,prog=self.layout_prog)
        elif self.layout_prog == 'dot':
            return self.position_dot(G)
        elif self.layout_prog == 'circular':
            return nx.drawing.layout.circular_layout(G)
        elif self.layout_prog == 'random':
            return nx.drawing.layout.random_layout(G)
        elif self.layout_prog == 'shell':
            return nx.drawing.layout.shell_layout(G)
        elif self.layout_prog == 'spring':
            return nx.drawing.layout.spring_layout(G)
        elif self.layout_prog == 'spectral':
            return nx.drawing.layout.spectral_layout(G)
        else:
            raise("Unknown Layout Program")
            

    def gen_faf_graphs(self, pno, numgens, threshold):
        """special utitlity function for graphing patents that interest us"""

        for name,values in faf.items():
            print(name,threshold)
            if threshold == -1:
                self.build_and_draw_pno_graph(values[0], numgens, name + ".pdf",
                                              threshold=values[1])
            else:
                self.build_and_draw_pno_graph(values[0], numgens, name + ".pdf",
                                              threshold=threshold)

    def generate_filename(self, ofile):
        # make filename
        patentname, file_extension = os.path.splitext(ofile)
        print(patentname)
        ofile = patentname + "_" + self.trait.trait_type + "_"
        if self.color_edges_by_parent:
            ofile += "parent_"
        else:
            ofile += "child_"
        if self.include_ancestors:
            ofile += "ancestors"
        if self.include_descendants:
            ofile += "descendants"
        ofile += file_extension
        return ofile

    # def generate_pca_model(self, wordvecs, ncomponents):
    #     """generate pca model for pca coloring"""

    #     pca = PCA(n_components=ncomponents)
    #     pca.fit(wordvecs)
    #     print("Explained Variance: " + str(pca.explained_variance_ratio_))
    #     return pca
