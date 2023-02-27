# Library:
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Constante:
NB_TRACE = 1000 # Nombre de trace souhaité
WIDTH = 100
HEIGHT = 100
L_MAX = 1
E_EPAISSEUR = 1 / 10
H_LIQUIDE = (1 / 10) - (1 / 100)
LONGUEUR_TRACE = 1

# Pour savoir si la norme est valide ou non:
def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

# Fonction pour déterminer si l'on se situe dans le cas1 (voir fichier txt):
def study_vector(z_1, z_2):
    if(z_1 < H_LIQUIDE and z_2 < H_LIQUIDE):
        return False
    elif(z_1 > (H_LIQUIDE + E_EPAISSEUR) and z_2 > (H_LIQUIDE + E_EPAISSEUR)):
        return False
    else:
        return True

# Fonction pour déterminer si l'on se situe dans le cas2,3 ou 4 (voir fichier txt):
def cas(lst_z):
    lst_z.sort()
    # Cas 2:
    if((lst_z[0] >= H_LIQUIDE 
        and lst_z[0] <= (H_LIQUIDE + E_EPAISSEUR)) 
        and (lst_z[1] >= H_LIQUIDE 
        and lst_z[1] <= (H_LIQUIDE + E_EPAISSEUR))): return 2
    # Cas 3:
    if (lst_z[0] <= H_LIQUIDE 
        and lst_z[1] >= H_LIQUIDE and lst_z[1] <= (H_LIQUIDE + E_EPAISSEUR)): return 3
    if (lst_z[0] <= (H_LIQUIDE + E_EPAISSEUR and lst_z[0] >= H_LIQUIDE) 
        and lst_z[1] >= (H_LIQUIDE + E_EPAISSEUR)): return 3
    # Cas 4:
    if(lst_z[0] <= H_LIQUIDE 
        and lst_z[1] >= (H_LIQUIDE + E_EPAISSEUR)): return 4

# Projeter vecteur sur le plan:
def projection(norme, cos_theta):
    return norme * cos_theta

# Fréquence des valeurs:
def frequency_val(lst):
    values, counts = np.unique(lst, return_counts=True)
    return dict(zip(values, counts))

# Matplotlib:
def plot_val(d):
    keys = []
    values = []
    items = d.items()
    for item in items:
        keys.append(item[0]), values.append(item[1])
    plt.plot(keys, values, 'ro-', linewidth=2, markersize=4)
    plt.show()

# Algorithme:
Lst = [] # Liste déterminer pour le plot
while len(Lst) < NB_TRACE:
    # Initialisation de l'origine du vecteur:
    x_1 = round(random.uniform(0, WIDTH), 2)
    y_1 = round(random.uniform(0, HEIGHT), 2)
    z_1 = round(random.uniform(0, 2 * H_LIQUIDE + E_EPAISSEUR + L_MAX), 2)
    # Initialisation des angles:
    phi = round(random.uniform(0, 2 * math.pi), 2)
    cos_theta = round(random.uniform(0, 1), 3)
    # Le vecteur:
    vector = np.array([
                    LONGUEUR_TRACE * cos_theta * math.cos(phi),
                    LONGUEUR_TRACE * cos_theta * math.sin(phi), 
                    LONGUEUR_TRACE * math.sin(math.acos(cos_theta))])
    # Verifier si l'on se situe dans le cas 1:
    if study_vector(z_1, z_1 + vector[2]):
        lst_x, lst_y, lst_z = [x_1, x_1 + vector[0]], \
            [y_1, y_1 + vector[1]], \
            [z_1, z_1 + vector[2]]
        lst_x.sort()
        lst_y.sort()
        # Projection en fonction des cas:
        # Cas 2:
        if cas(lst_z) == 2: 
            norme_p = projection(LONGUEUR_TRACE, cos_theta)
            Lst.append(round(norme_p, 5))
        # Cas 3:
        if cas(lst_z) == 3: 
            if lst_z[0] < H_LIQUIDE:
                x_3 = vector[0] - \
                (H_LIQUIDE / LONGUEUR_TRACE *\
                     math.sin(math.acos(cos_theta))) * vector[0]
                y_3 = vector[1] - \
                (H_LIQUIDE / LONGUEUR_TRACE *\
                     math.sin(math.acos(cos_theta))) * vector[1]
                norme_p = math.sqrt(math.pow(x_3, 2 ) + math.pow(y_3, 2))
                # Lst.append(round(norme_p, 2))
            else:
                x_3 = vector[0] - ((H_LIQUIDE + E_EPAISSEUR)\
                / LONGUEUR_TRACE * math.sin(math.acos(cos_theta))) * vector[0]
                y_3 = vector[1] - ((H_LIQUIDE + E_EPAISSEUR)\
                / LONGUEUR_TRACE * math.sin(math.acos(cos_theta))) * vector[1]
                norme_p = math.sqrt(math.pow(x_3, 2 ) + math.pow(y_3, 2)) 
                # Lst.append(round(norme_p, 2))
        # Cas 4:
        if cas(lst_z) == 4:
            norme_p = projection(E_EPAISSEUR, cos_theta)
            # Lst.append(round(norme_p, 2))
        # Vérification si une projection est supérieur à la longueur de la norme:
        # if norme_p > LONGUEUR_TRACE:
        #     print("_____PROJECTION INVALIDE_____")
        #     print(colored(255,0,0, norme_p))
        # else:
        #     Lst.append(round(norme_p, 2))

# Final Plot
dic = frequency_val(Lst)
plot_val(dic)
