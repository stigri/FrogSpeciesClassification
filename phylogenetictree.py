from ete3 import Tree, TreeStyle, Tree, TextFace, add_face_to_node, ImgFace, NodeStyle

Megophrys = '(nasuta:1)Megophrys:1'
Leptobrachium = '(hasseltii:1)Leptobrachium:1'
Megophridae = '({},{})Megophridae:4'.format(Megophrys, Leptobrachium)

Ingerophrynus = '(Ansonia:1, (parvus:1)Ingerophrynus:1'
Leptophryne = '(borbonica:1)Leptophryne:1'
Pelophryne = '(signata:1)Pelophryne:1'
Phrynoidis = '(asper:1,juxtasper:1)Phrynoidis:1'
Bufonidae = '({},{},{},{})Bufonidae:7'.format(Ingerophrynus, Leptophryne, Pelophryne, Phrynoidis)

Microhyla = '(achatina:1,heymonsi:1)Microhyla:1'
Microhylidae = '(({})Microhylidae:1'.format(Microhyla)

Fejervarja = '(limnocharis:1)Fejervarja:1'
Limnonectes = '(macrodon:1, paramacrodon:1, kuhlii:1,hikidai:1,blythii:1,sisikdagu:1)Limnonectes:1'
Occidozyga = '(sumatrana:1)Occidozyga:1'
Dicroglassidae = '(({},{},{})Dicroglassidae:1'.format(Fejervarja, Limnonectes, Occidozyga)

Sumaterana = '(montana:1,crassiovis:1, dabulescens:1)Sumaterana:1'
Pulchrana = '(picturata:1,glandulosa:1,rawa:1,debussyi:1)Pulchrana:1'
Odorrana = '(hosii:1)Odorrana:1'
Amnirana = '(nicobariensis:1)Amnirana:1'
Chalcorana = '(rufipes:1,chalconota:1)Chalcorana:1'
Huia = '(masonii:1)Huia:1'
Hylarana = '(erythraea:1)Hylarana:1'
Ranidae = '(({},{},{},{},{},{},{})Ranidae:1'.format(Sumaterana, Pulchrana, Odorrana, Amnirana, Chalcorana, Huia, Hylarana)

Philautus = 'Philautus:1'
Polypedates = '(macrotis:1, otilophus:1,leucomystax:1)Polypedates:1'
Rhacophorus = '(poecilonotus:1,margaritifer:1,catamitus:1,cyanopunctatus:1,prominanus:1)Rhacophorus:1'
Nyctixalus = '(pictum:1)Nyctixalus:1'
Rhacophoridae = '({},{},{},{})Rhacophoridae:2'.format(Philautus, Polypedates, Rhacophorus, Nyctixalus)


t = Tree(
    "(({},{},{},{},{},{})a:1)a:5)a:2)a:2)a:5)Anura;".format(Megophridae, Bufonidae, Microhylidae, Dicroglassidae, Ranidae, Rhacophoridae),
    format=1)

ns = NodeStyle()
ns["fgcolor"] = "#000000"
ns["shape"] = "circle"
ns["vt_line_width"] = 1
ns["hz_line_width"] = 1
ns["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
ns["hz_line_type"] = 1


ts = TreeStyle()
ts.show_leaf_name = False
ts.branch_vertical_margin = 15


def my_layout(node):
    F = TextFace(node.name, tight_text=True)
    if node.name is not 'a':
        add_face_to_node(F, node, column=0, position="branch-right")
    # if node.is_leaf():
    #     img = ImgFace('/home/stine/Desktop/{}.jpg'.format(node.name))
    #     add_face_to_node(img, node, column=0, position='aligned')
    for n in t.iter_search_nodes():
        if n.dist > 1:
            n.img_style = ns





ts.layout_fn = my_layout
t.show(tree_style=ts)
