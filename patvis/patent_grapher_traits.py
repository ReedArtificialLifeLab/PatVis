# This file defines the trait interface to the PatentGrapher class.
# The PatentGrapher class is structured similarly to the gpe code.
# Traits are abstracted out of the main file (here, patent_grapher.py)
# to make it easy to generate visualizations with different trait types.

# To visualize a new trait, simply make a class that implements the attributes
# and functions of the TraitVisual class found below and then add the class to the dispatch
# dictionary at the end of the file.

class TraitVisual(object):
    def __init__(self):
        # query criteria for db searches
        # all patents in visualization must match these criteria
        self.criteria = {}
        # the fields to be returned from the database
        self.projection = {}
        self.trait_type = ''
        # the db collection to look in for patent documents
        self.collection = '' 
        # the returned field that holds the patent number
        self.pno_field = ''
        # returned field listing patent numbers that this patent cites
        self.cited = ''
        # the returned field listing patent numbers that cite this patent
        self.citedby = ''
    def generate_color_scheme():
        """
        generate a color scheme based on the Graph G and 
        the color scheme from the last generation.
        the previous color scheme will always be None unless
        draw_each_gen = True
        """
        raise("TraitVisual is abstract")
    def get_node_attributes():
        """
        defines the info to store in each nodes attributes dictionary.
        you can define graph viz node attributes here if you'd like to 
        modify how the nodes are drawn.
        """
        raise("TraitVisual is abstract")

from patvis import get_class_to_cat_dict,get_class_to_subcat_dict,get_category_descriptions_dict,get_subcategory_descriptions_dict, get_class_descriptions, get_cluster_descriptions
from patvis.discrete_color import discrete_color_scheme

_class_to_cat = get_class_to_cat_dict()
_class_to_subcat = get_class_to_subcat_dict()
describe_cat = get_category_descriptions_dict()
describe_subcat = get_subcategory_descriptions_dict()
patentclasses = get_class_descriptions()
patentclusters = get_cluster_descriptions()

def hex_to_rgb(v):
    v = v.lstrip('#')
    lv = len(v)
    return tuple(int(v[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    r,g,b = [int(x*255) for x in rgb]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def grayscale(self,rgb):
    r, g, b = rgb
    lum_avg = 0.21*r + 0.72*g + 0.07*b
    return rgb_to_hex((lum_avg,lum_avg,lum_avg))

class IssueDateUSPCClass(TraitVisual):
    def __init__(self):
        self.criteria = { 'rawcites' : {'$exists' : True }, 'citedby' : {'$exists' : True }, 'mainUSClass' : {'$exists' : True }, 'isd' : {'$exists' : True } }
        self.projection = { '_id' : 1, 'citedby' : 1, 'rawcites' : 1, 'mainUSClass' : 1, 'isd' : 1 }
        self.trait_type = 'issue class'
        self.collection = 'traits'
        self.pno_field = '_id'
        self.cited = 'rawcites'
        self.citedby = 'citedby'

    def generate_color_scheme(self, G, color_scheme):
        # make a list of 10 hex colors equally spaced around the colorwheel
        colorvals = list(map(rgb_to_hex,discrete_color_scheme(n=10)))
        # count the occurrences of each class in the graph
        class_counts = defaultdict(int)
        for node in G.nodes_iter():
            class_counts[ node[self.trait_type] ] += 1
        # get a list of the top ten most frequent classes in the graph
        topcolors = sorted(class_counts, key=class_counts.get, reverse=True)[:9]

        # if there is no previous color scheme, generate one based on the top classes
        # if there is, fill it out with any classes that have been added to the graph
        if color_scheme == None:
            color_scheme = { topclass : color for topclass,color in zip(topcolors,colorvals) }
        elif len(color_scheme) < 9:
            for c in topcolors:
                if len(color_scheme) >= 9:
                    break
                if c not in color_scheme:
                    color_scheme[c] = colorvals[len(color_scheme)]
        return color_scheme
    
    def get_node_attributes(self, patent):
        return { 'issue class' : patent.get('mainUSClass',"Abolished Class"),
                 'label' : patentclasses.get(patent['mainUSClass'],"Abolished Class") + "\n" + patent['mainUSClass'] + "\n" + str(patent[self.pno_field]),
                 'style' : 'filled',
                 'shape' : 'box',
                 'year' : patent['isd'].year }

class NBERCategory(TraitVisual):
    def __init__(self):
        self.criteria = { 'rawcites' : {'$exists' : True }, 'citedby' : {'$exists' : True }, 'mainUSClass' : {'$exists' : True }, 'isd' : {'$exists' : True } }
        self.projection = { '_id' : 1, 'citedby' : 1, 'rawcites' : 1, 'mainUSClass' : 1, 'isd' : 1 }
        self.trait_type = 'category'
        self.collection = 'traits'
        self.pno_field = '_id'
        self.cited = 'rawcites'
        self.citedby = 'citedby'
        
    def generate_color_scheme(self, G, color_scheme):
        # if self.grayscale:
        #     colorvals = list(map(rgb_to_hex,[ ((.9/5)*i,(.9/5)*i,(.9/5)*i) for i in range(6) ]))
        # else:
        color_values = list(map(rgb_to_hex,discrete_color_scheme(n=6)))
        return { cat+1 : color for cat,color in enumerate(color_values) }
        
    def get_node_attributes(self, patent):
        category = self.class_to_category(patent.get('mainUSClass',None))
        return { 'category' : category,
                 'label' : describe_cat.get(category,'No Category') + "\n" + str(patent[self.pno_field]),
                 'style' : 'filled',
                 'shape' : 'box',
                 'height' : 2,
                 'width' : 2,
                 'penwidth' : 0,
                 'year' : patent['isd'].year }        

    def class_to_category(self,mainclass):
        """find NBER category based on US Tech Class"""

        return _class_to_cat.get(mainclass,None)

class NBERSubcategory(TraitVisual):
    def __init__(self):
        self.criteria = { 'rawcites' : {'$exists' : True }, 'citedby' : {'$exists' : True }, 'mainUSClass' : {'$exists' : True }, 'isd' : {'$exists' : True } }
        self.projection = { '_id' : 1, 'citedby' : 1, 'rawcites' : 1, 'mainUSClass' : 1, 'isd' : 1 }
        self.trait_type = 'subcategory'
        self.collection = 'traits'
        self.pno_field = '_id'
        self.cited = 'rawcites'
        self.citedby = 'citedby'
        
    def generate_color_scheme(self, G, color_scheme):
        color_values = list(map(rgb_to_hex,discrete_color_scheme(n=36)))
        return { subcat+1 : color for subcat,color in enumerate(color_values) }
        
    def get_node_attributes(self, patent):
        subcategory = self.class_to_subcategory(patent.get('mainUSClass',None))
        attrib = { 'subcategory' : subcategory,
                   'label' : describe_subcat.get(subcategory,'No Subcategory') + "\n" + str(patent[self.pno_field]),
                   'style' : 'filled',
                   'shape' : 'box',
                   'year' : patent['isd'].year }
        
    def class_to_subcategory(self,mainclass):
        """find NBER subcategory based on US Tech Class"""

        return _class_to_subcat.get(mainclass,None)

class W2VCluster(object):
    def __init__(self):
        self.criteria = { 'rawcites' : {'$exists' : True }, 'citedby' : {'$exists' : True }, 'wordvec_clusters' : {'$exists' : True }, 'isd' : {'$exists' : True } }
        self.projection = { '_id' : 1, 'citedby' : 1, 'rawcites' : 1, 'isd' : 1, 'wordvec_clusters' : { '$slice' : 4 } }
        self.trait_type = 'w2v'
        self.collection = 'traits'
        self.pno_field = '_id'
        self.cited = 'rawcites'
        self.citedby = 'citedby'
        
    def generate_color_scheme(self, G, color_scheme):
        raise("W2VCluster does not implement generate_color_scheme")
    
    def get_node_attributes(self, patent):
        return { 'w2vcluster' : patent.get(patent['wordvec_clusters'][0],"Corrupted Cluster ID"),
                 'label' : patentclusters.get(patent['wordvec_clusters'][0],"Corrupted Cluster ID") + "\n" + str(patent['wordvec_clusters'][0]) + "\n" + str(patent[self.pno_field]),
                 'style' : 'filled',
                 'shape' : 'box',
                 'year' : patent['isd'].year }

class CurrentUSPCClass(TraitVisual):
    def __init__(self):
        self.criteria = { 'rawcites' : {'$exists' : True }, 'citedby' : {'$exists' : True }, 'uspc_current_main_class' : {'$exists' : True }, 'issue_year' : {'$exists' : True } }
        self.projection = { '_id' : 1, 'citedby' : 1, 'rawcites' : 1, 'uspc_current_main_class' : 1, 'isd' : 1 }
        self.trait_type = 'current class'
        self.collection = 'patns'
        self.pno_field = 'pno'
        self.cited = 'rawcites'
        self.citedby = 'citedby'
        
    def generate_color_scheme(self, G, color_scheme):
        # make a list of 10 hex colors equally spaced around the colorwheel
        colorvals = list(map(rgb_to_hex,discrete_color_scheme(n=10)))
        # count the occurrences of each class in the graph
        class_counts = defaultdict(int)
        for node in G.nodes_iter():
            class_counts[ node[self.trait_type] ] += 1
        # get a list of the top ten most frequent classes in the graph
        topcolors = sorted(class_counts, key=class_counts.get, reverse=True)[:9]

        # if there is no previous color scheme, generate one based on the top classes
        # if there is, fill it out with any classes that have been added to the graph
        if color_scheme == None:
            color_scheme = { topclass : color for topclass,color in zip(topcolors,colorvals) }
        elif len(color_scheme) < 9:
            for c in topcolors:
                if len(color_scheme) >= 9:
                    break
                if c not in color_scheme:
                    color_scheme[c] = colorvals[len(color_scheme)]
        return color_scheme
    
    def get_node_attributes(self, patent):
        return { 'current class' : patent.get('uspc_current_main_class',None),
                 'label' : patentclasses.get(patent['uspc_current_main_class'],"Missing Class") + "\n" + patent['uspc_current_main_class'] + "\n" + str(patent[self.pno_field]),
                 'style' : 'filled',
                 'shape' : 'box',
                 'year' : patent['isd'].year }
    
traits = {
    'category' : NBERCategory,
    'subcategory' : NBERSubcategory,
    'current class' : CurrentUSPCClass,
    'issue class' : IssueDateUSPCClass,
    'w2v cluster' : W2VCluster
}
    
def _get_trait(trait):
    return traits[trait]()
