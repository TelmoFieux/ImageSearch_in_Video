# ImageSearch_in_Video
Ce programme en python permet de scanner une vidéo pour tenter de trouver une image au sein de celle ci qui correspond a une image que l'utilisateur cherche. Pour cela vous pouvez importer vos propres images ou faire une recherche d'image via l'API de google. Si vous le faites il faudra paramétré le nombre d'image récupéré, la recherche effectué, mais aussi paramétré votre propre clé et search engine.

Pour ce qui est du fonctionnement du programme en lui même, il parcours la vidéo et extrait une image toute les X frames. Cette image et comparé au images fourni par l'utilisateur grâce à un model d'IA. Si le seuil de ressemblance est passé, alors on enregistre le timestamp. Le seuil de ressemblance ainsi que la fréquence d'analyse sont paramétrable. A la fin sont affiché tout les timestamp qui ont passé le seuil de ressemblance avec le coefficient de ressemblance affiché.
J'ai implémenté une deuxième méthode un peu moins "bruttte de décoffrage" que j'ai appelé la file de voiture et qui suis ce principe:

Cette méthode part du principe que plus on se rapproche d'une séquence de la vidéo ou se trouve l'élément recherché, plus le coeffictient de ressemblance augmentera. Par exemple si je cherche une séquence aquatique dans ma vidéo et que donc mes image de références ont beaucoup de teinte bleu, plus je me repproche de cette séquence plus les coefficient de ressemblance seront élevé. On va donc appliqué des seuil arbitraire pour lesquels il faudra effectué des test pour trouver le "sweet spot". Dans le programme se trouve ma configuration qui a plutôt bien marché pour moi mais on peut surement trouver mieux. Par exemple si le coefficent de ressemblance detecté est 50% on va passer le frame_gap a 10 sec au lieu de 5sec car on est probablement loin de la séquence recherché. A l'inverse, si le coefficient augmente par exemple 80, on va passer de 5 a 2 sec. Voici une petite illustration:

thread 1   Thread 1	Thread 1	Thread 1       Thread 1 detecte une proba élévé	          thread 1   Thread 1	Thread 1	Thread 1
   300	      600 	   900		   1200		-------------------------------->	 	              960	      1080     1320		   1440
		{Check toutes les 5 sec}							{check toute les 2 sec par exemple
														              l'important c'est qu'on ralentit}

Enfin, pour s'assurer qu'on ne rate rien au cas ou on a "freiné" trop vite, on va revenir légèrement en arrière avec le nouveau frame_gap.
Dans cette exemple on check toute les frame entre la frame trigger et la précédente avec le nouvel intervalle ici 2. On verifiera alors 2 frame en arrière avant de continuer. Attention les frames arrière qu'on check ne modifie pas la vitesse de la file.
Cela permet donc d'assurer qu'on a pas sauté le passage recherché et coûte peu cher en puissance de calcul. On rajoute entre 1 et 2 frame a vérifier. L'important est de trouver une bonne configuration pour éviter de trop "freiner".
Voici un exemple de benchemark avec une video de 1h54 sur mon PC portable avec un Processeur ryzen 5600 with integrated gpu pour montrer les différence de performance : 

Test 1 : 2min48sec  check toute les 5 secondes pour la vidéo "Ratchet_et_clank_2_part_7_CRF.mp4" de 1h54
matchin_images_treshold = 88 | num_requests = 2 | filtering_treshold = 0.80 | frame_gap = 300 | fonction utilisé : V2 avec gpu | batche_size=1
6 image de reference + extracted frame

Résultats : image found at timestamps: {10.416666666666666: 88.75721096992493, 10.5: 89.77494239807129, 10.583333333333334: 90.58698415756226, 56.5: 92.62803792953491}

=============================================================================================================================================================================

Changements : implémentation de la vitesse adaptative en fonction du pourcentage de ressemblance

Test 2 : 1min08sec  seuil={75:20,80:10,85:5,90:2,100:2}  pour la vidéo "Ratchet_et_clank_2_part_7_CRF.mp4" de 1h54
matchin_images_treshold = 88 | num_requests = 2 | filtering_treshold = 0.80 | fonction utilisé : V2 avec gpu | batche_size=1 
6 image de reference + extracted frame

Résultats : image found at timestamps: {10.583333333333334: 90.63405990600586}

=============================================================================================================================================================================

Changements : implémentation de la file de voiture (on check les frames précédentes au moment ou on ralentit pour être sur d'avoir rien raté)

Test 3 : 1min27sec  seuil={75:20,80:10,85:5,90:2,100:2}  pour la vidéo "Ratchet_et_clank_2_part_7_CRF.mp4" de 1h54
matchin_images_treshold = 88 | num_requests = 2 | filtering_treshold = 0.80 | fonction utilisé : V2 avec gpu | batche_size=1 | programme_recherche_intelligent
6 image de reference + extracted frame

Résultats : image found at timestamps: {10.5: 89.82644081115723, 10.383333333333333: 89.16247487068176, 10.433333333333334: 88.39675188064575, 10.55: 88.79101276397705, 10.583333333333334: 90.63405990600586}

/*========================================================================================================================================================================================*/

On constate donc que la méthode de la file de voiture est plus rapide et plus précise avec les bon paramètre. Voici un exemple de répartition des image en fonction des seuil pour donner une idée de comment ça marche:

Pour la recherche Ratchet and Clank ps2 hydropack sur la vidéo Ratchet_et_clank_2_part_7_CRF.mp4 avec 5 sec de frame gap:

- 0.0'%' d'image entre 0'%' et 20'%' de similarité
- 0.0'%' d'image entre 20'%' et 40'%' de similarité
- 0.07309941520467836'%' d'image entre 40'%' et 60'%' de similarité
- 31.432748538011694'%' d'image entre 60'%' et 80'%' de similarité
- 68.42105263157895'%' d'image entre 80'%' et 100'%' de similarité
- 168 accélérations
- 170 décélération
time : 168 sec

C'est a peu près tout pour ce programme. Il est toujours possible de modifier des élément pour le rendre plus performant. Changer de modèle, changer le batch_size en fonction de la puissance de votre gpu etc. A noter que j'ai développé ce programme sur un gpu amd je ne sais pas si cela fonctionne sur un gpu nvidia.
