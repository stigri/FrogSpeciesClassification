from ete3 import Tree, TreeStyle, Tree, TextFace, add_face_to_node, ImgFace, NodeStyle, Face, TreeFace
import utils

Megophrys = Tree('(nasuta)Megophrys;', format=1)
Leptobrachium = Tree('(hasseltii)Leptobrachium;', format=1)
# Megophridae = Tree('Megophridae:4;')
# Megophridae.add_child(Megophrys)
# Megophridae.add_child(Leptobrachium)
# print(Megophridae)
Megophridae = 'Megophridae:4'

Ansonia = Tree('Ansonia;', format=1)
Ingerophrynus = Tree('(parvus)Ingerophrynus;', format=1)
Leptophryne = Tree('(borbonica)Leptophryne;', format=1)
Pelophryne = Tree('(signata)Pelophryne;', format=1)
Phrynoidis = Tree('(asper,juxtasper)Phrynoidis;', format=1)
# Bufonidae = Tree('Bufonidae:7;', format=1)
# Bufonidae.add_child(Ansonia)
# Bufonidae.add_child(Ingerophrynus)
# Bufonidae.add_child(Leptophryne)
# Bufonidae.add_child(Pelophryne)
# Bufonidae.add_child(Phrynoidis)
# print(Bufonidae)
Bufonidae = 'Bufonidae:7'

Microhyla = Tree('(achatina,heymonsi)Microhyla;', format=1)
# Microhylidae = Tree('Microhylidae:1;', format=1)
# Microhylidae.add_child(Microhyla)
# print(Microhylidae)
Microhylidae = 'Microhylidae:1'

Fejervarja = Tree('(limnocharis)Fejervarja;', format=1)
Limnonectes = Tree('(macrodon, paramacrodon, kuhlii,hikidai,blythii,sisikdagu)Limnonectes;', format=1)
Occidozyga = Tree('(sumatrana)Occidozyga;', format=1)
# Dicroglassidae = Tree('Dicroglassidae:1;', format=1)
# Dicroglassidae.add_child(Fejervarja)
# Dicroglassidae.add_child(Limnonectes)
# Dicroglassidae.add_child(Occidozyga)
# print(Dicroglassidae)
Dicroglassidae = 'Dicroglassidae:1'

Sumaterana = Tree('(montana:1,crassiovis, dabulescens)Sumaterana;', format=1)
Pulchrana = Tree('(picturata,glandulosa,rawa,debussyi)Pulchrana;', format=1)
Odorrana = Tree('(hosii)Odorrana;', format=1)
Amnirana = Tree('(nicobariensis)Amnirana;', format=1)
Chalcorana = Tree('(rufipes,chalconota)Chalcorana;', format=1)
Huia = Tree('(masonii)Huia;', format=1)
Hylarana = Tree('(erythraea)Hylarana;', format=1)
# Ranidae = Tree('Ranidae:1;', format=1)
# Ranidae.add_child(Sumaterana)
# Ranidae.add_child(Pulchrana)
# Ranidae.add_child(Odorrana)
# Ranidae.add_child(Amnirana)
# Ranidae.add_child(Chalcorana)
# Ranidae.add_child(Huia)
# Ranidae.add_child(Hylarana)
# print(Ranidae)
Ranidae = 'Ranidae:1'

Philautus = Tree('Philautus;', format=1)
Polypedates = Tree('(macrotis, otilophus,leucomystax)Polypedates;', format=1)
Rhacophorus = Tree('(poecilonotus,margaritifer,catamitus,cyanopunctatus,prominanus)Rhacophorus;', format=1)
Nyctixalus = Tree('(pictum)Nyctixalus;', format=1)
# Rhacophoridae = Tree('Rhacophoridae:2;', format=1)
# Rhacophoridae.add_child(Philautus)
# Rhacophoridae.add_child(Polypedates)
# Rhacophoridae.add_child(Rhacophorus)
# Rhacophoridae.add_child(Nyctixalus)
# print(Rhacophoridae)
Rhacophoridae = 'Rhacophoridae:2'

fork1 = Tree('a:5;', format=1)
fork2 = Tree('a:2;', format=1)
fork3 = Tree('a:2;', format=1)
fork4 = Tree('a:5;', format=1)
fork5 = Tree('a:1;', format=1)

Anura = Tree('Anura;', format=1)

#
# fork5.add_child(Ranidae)
# fork5.add_child(Rhacophoridae)
# fork4.add_child(Dicroglassidae)
# fork4.add_child(fork5)
# fork3.add_child(Microhylidae)
# fork3.add_child(fork4)
# fork2.add_child(Bufonidae)
# fork2.add_child(fork3)
# fork1.add_child(Megophridae)
# fork1.add_child(fork2)
# Anura.add_child(fork1)
# print(fork1)

# Megophrys = '(nasuta:1)Megophrys:1'
# Leptobrachium = '(hasseltii:1)Leptobrachium:1'
# Megophridae = '({},{})Megophridae:4'.format(Megophrys, Leptobrachium)
#
# Ingerophrynus = '(Ansonia:1, (parvus:1)Ingerophrynus:1'
# Leptophryne = '(borbonica:1)Leptophryne:1'
# Pelophryne = '(signata:1)Pelophryne:1'
# Phrynoidis = '(asper:1,juxtasper:1)Phrynoidis:1'
# Bufonidae = '({},{},{},{})Bufonidae:7'.format(Ingerophrynus, Leptophryne, Pelophryne, Phrynoidis)
#
# Microhyla = '(achatina:1,heymonsi:1)Microhyla:1'
# Microhylidae = '(({})Microhylidae:1'.format(Microhyla)
#
# Fejervarja = '(limnocharis:1)Fejervarja:1'
# Limnonectes = '(macrodon:1, paramacrodon:1, kuhlii:1,hikidai:1,blythii:1,sisikdagu:1)Limnonectes:1'
# Occidozyga = '(sumatrana:1)Occidozyga:1'
# Dicroglassidae = '(({},{},{})Dicroglassidae:1'.format(Fejervarja, Limnonectes, Occidozyga)
#
# Sumaterana = '(montana:1,crassiovis:1, dabulescens:1)Sumaterana:1'
# Pulchrana = '(picturata:1,glandulosa:1,rawa:1,debussyi:1)Pulchrana:1'
# Odorrana = '(hosii:1)Odorrana:1'
# Amnirana = '(nicobariensis:1)Amnirana:1'
# Chalcorana = '(rufipes:1,chalconota:1)Chalcorana:1'
# Huia = '(masonii:1)Huia:1'
# Hylarana = '(erythraea:1)Hylarana:1'
# Ranidae = '(({},{},{},{},{},{},{})Ranidae:1'.format(Sumaterana, Pulchrana, Odorrana, Amnirana, Chalcorana, Huia, Hylarana)
#
# Philautus = 'Philautus:1'
# Polypedates = '(macrotis:1, otilophus:1,leucomystax:1)Polypedates:1'
# Rhacophorus = '(poecilonotus:1,margaritifer:1,catamitus:1,cyanopunctatus:1,prominanus:1)Rhacophorus:1'
# Nyctixalus = '(pictum:1)Nyctixalus:1'
# Rhacophoridae = '({},{},{},{})Rhacophoridae:2'.format(Philautus, Polypedates, Rhacophorus, Nyctixalus)
#


t = Tree(
    "(({},({},({},({},({},{})a:1)a:5)a:2)a:2)a:5)Anura;".format(Megophridae, Bufonidae, Microhylidae, Dicroglassidae,
                                                                Ranidae, Rhacophoridae),
    format=1)



# ns = NodeStyle()
# ns["fgcolor"] = "black"
# ns["size"] = 5
# n1 = (t&'borbonica')
# n2 = (t&'borbonica').up
# n1.set_style(nst1)
# n2.set_style(nst1)


families = {'Megophridae': '(Litter Frogs)', 'Bufonidae': '(True Toads)', 'Microhylidae': '(Narrow-mouthed Frogs)',
            'Dicroglassidae': '(True Frogs I)', 'Ranidae': '(True Frogs II)',
            'Rhacophoridae': '(Afro-asian Tree Frogs)'}
genera_trees = [[Megophrys, Leptobrachium], [Ansonia, Ingerophrynus, Leptophryne, Pelophryne, Phrynoidis], [Microhyla],
                [Fejervarja, Limnonectes, Occidozyga],
                [Sumaterana, Pulchrana, Odorrana, Amnirana, Chalcorana, Huia, Hylarana],
                [Philautus, Polypedates, Rhacophorus, Nyctixalus]]
genera = [['Megophrys', 'Leptobrachium'], ['Ansonia', 'Ingerophrynus', 'Leptophryne', 'Pelophryne', 'Phrynoidis'],
          ['Microhyla'],
          ['Fejervarja', 'Limnonectes', 'Occidozyga'],
          ['Sumaterana', 'Pulchrana', 'Odorrana', 'Amnirana', 'Chalcorana', 'Huia', 'Hylarana'],
          ['Philautus', 'Polypedates', 'Rhacophorus', 'Nyctixalus']]

dict_genera = {'Megophrys': '(Horned Frogs)', 'Leptobrachium': '(Litter Frogs)',
          'Ansonia': '(Stream Toads)', 'Ingerophrynus': '(Asian Forest Toads)',
           'Leptophryne': '(Indenesian Tree Toads)', 'Pelophryne': '(Dwarf Toads)', 'Phrynoidis': '(River Toads)',
          'Microhyla': '(Narrow-mouthed Frogs)',
          'Fejervarja': '(Terrestrial Frogs)', 'Limnonectes': '(Fanged Frogs)', 'Occidozyga': '(Puddle Frogs)',
          'Sumaterana': '(Cascade Frogs)', 'Pulchrana': '(Asian Ranid Frogs)', 'Odorrana': '(Smelling Frogs)',
           'Amnirana': '(Asian Ranid Frogs)', 'Chalcorana': '(White-lipped Ranid Frogs)', 'Huia': '(Cascade Frogs)',
           'Hylarana': '(White-lipped Frogs)',
          'Philautus': '(Bush Frogs)', 'Polypedates': '(Whipping Frogs)', 'Rhacophorus': '(Parachuting Frogs)',
           'Nyctixalus': '(Indonesian Tree Frogs)'}

dict_species = {'nasuta': '(Bornean Horned Frog', 'hasseltii': '(Hasselt\'s Toad)', 'parvus': '(Lesser Toad)',
           'borbonica': '(Cross Toad)', 'signata': '(Lowland Dwarf Toad)', 'asper': '(Asian Giant Toad)',
           'juxtasper': '(Giant River Toad)', 'achatina': '(Javan Chorus Frog)', 'heymonsi': '(Dark-sided Chorus Frog)',
           'limnocharis': '(Grass Frog)', 'macrodon': '(Fanged River Frog)', 'kuhlii': '(Kuhl\'s Creek Frog)',
           'hikidai': '(Rivulet Frog)', 'blythii': '(Blyth\'s Frog )', 'paramacrodon': '(Lesser Swamp Frog)',
           'sumatrana': '(Sumatran Puddle Frog)', 'montana': '(Mountain Cascade Frog)',
           'dabulescens': '(Gayo Cascade Frog)', 'picturata': '(Spotted Stream Frog)',
           'glandulosa': '(Rough-sided Frog)', 'hosii': '(Poisonous Rock Frog)', 'nicobariensis': '(Cricket Frog)',
           'chaloconota': '(Brown Stream Frog)', 'masonii': '(Javan Torrent Frog)', 'erythraea': '(Green Paddy Frog)',
           'macrotis': '(Dark-eared Tree Frog)', 'otilophus': '(File-eared Tree Frog)',
           ' leucomystax': '(Four-lined Tree Frog)', 'poecilonotus': '(Sumatra Flying Frog)',
           'margaritifer': '(Java Flying Frog)', 'cyanopunctatus': '(Blue-spotted Bush Frog)',
           'prominanus': '(Johore Flying Frog)', 'pictum': '(Peter\'s Tree Frog)'}
colors = {'Megophrys': 'olive', 'Leptobrachium': 'gold', 'Ansonia': 'lavender', 'Ingerophrynus': 'navy',
          'Leptophryne': 'maroon', 'Pelophryne': 'pink', 'Phrynoidis': 'magenta', 'Microhyla': 'red',
          'Fejervarja': 'orange', 'Limnonectes': 'cyan', 'Occidozyga': 'deeppink', 'Sumaterana': 'sienna',
          'Pulchrana': 'darkorchid', 'Odorrana': 'purple', 'Amnirana': 'plum', 'Chalcorana': 'lime', 'Huia': 'tomato',
          'Hylarana': 'gray', 'Philautus': 'green', 'Polypedates': 'yellow', 'Rhacophorus': 'teal',
          'Nyctixalus': 'darkkhaki'}

ns = NodeStyle()
ns["fgcolor"] = "black"
ns["shape"] = "circle"
ns["vt_line_width"] = 1
ns["hz_line_width"] = 1
ns["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
ns["hz_line_type"] = 1
ns['size'] = 5
ns_genera = NodeStyle()
ns["fgcolor"] = "black"
ns["shape"] = "circle"
ns['size'] = 5
ns["vt_line_width"] = 1
ns["hz_line_width"] = 1
ns["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
ns["hz_line_type"] = 0

ts_genera = TreeStyle()
ts = TreeStyle()

def my_layout(node):
    F = TextFace(node.name, tight_text=True, penwidth=5, fsize=12)
    if node.name is not 'a':
        add_face_to_node(F, node, column=0, position="branch-right")
        node.set_style(ns)
        if node.name in families:
            G = TextFace(families[node.name], fstyle='italic')
            add_face_to_node(G, node, column=0, position="branch-right")
            node.set_style(ns)
    if node.name in dict_genera:
        H = TextFace(dict_genera[node.name], fstyle='italic')
        add_face_to_node(H, node, column=0, position="branch-right")
        node.set_style(ns)
    if node.name in dict_species:
        I = TextFace(dict_species[node.name], fstyle='italic')
        add_face_to_node(I, node, column=0, position="branch-right")
        node.set_style(ns)

    # if node.is_leaf():
    # node.img_style["size"] = 0
    # img = ImgFace('/home/stine/Desktop/{}.jpg'.format(node.name))
    # add_face_to_node(img, node, column=0, aligned=True)
    for n in Anura.iter_search_nodes():
        if n.dist > 1:
            n.img_style = ns


for idx, family in enumerate(families):
    for cidx, genus_tree in enumerate(genera_trees[idx]):
        tf_genera = TreeFace(genus_tree, ts_genera)
        tf_genera.border.width = 2
        genus = genera[idx][cidx]
        color = colors[str(genus)]
        tf_genera.border.color = color
        (t & family).add_face(tf_genera, column=0, position='aligned')

for n in genus_tree.iter_search_nodes():
    if n.dist == 1:
        n.img_style = ns_genera



ts_genera.show_leaf_name = False
ts_genera.show_scale = False
ts_genera.layout_fn = my_layout
ts.branch_vertical_margin = 10

ts.show_leaf_name = False
ts.branch_vertical_margin = 15
ts.layout_fn = my_layout
ts.draw_guiding_lines = True
ts.guiding_lines_type = 2
ts.show_scale = False
ts.allow_face_overlap = False
# ts.mode = "c"
# ts.arc_start = 180 # 0 degrees = 3 o'clock
# ts.arc_span = 270
t.show(tree_style=ts)
t.render("mytree.png", w=183, units="mm", tree_style=ts)
