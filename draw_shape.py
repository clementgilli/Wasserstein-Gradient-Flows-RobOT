import cv2
import numpy as np

# --- CONFIGURATION ---
CANVAS_SIZE = 800  # Un peu plus grand pour y voir clair sur Retina
LINE_THICKNESS = 15
FILENAME_A = "data/shape_a.png"
FILENAME_B = "data/shape_b.png"

# --- ETAT GLOBAL ---
drawing = False
prev_pt = None  # Stocke le point précédent (x, y)
current_mode = 1  # 1: Source, 2: Target

# Flags pour optimiser le rafraîchissement
dirty_canvas = True  # L'image a changé
dirty_overlay = True # L'overlay (onion skin) doit être recalculé

# Images de base (Blanc = 255)
img_A = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
img_B = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255

# Cache pour l'affichage (évite de tout recalculer à chaque frame)
display_buffer = None

def mouse_callback(event, x, y, flags, param):
    global drawing, prev_pt, img_A, img_B, dirty_canvas, dirty_overlay

    target_img = img_A if current_mode == 1 else img_B

    # CLIC ENFONCÉ : DÉBUT D'UN NOUVEAU TRAIT (DISCONTINU)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_pt = (x, y)
        # On dessine un point immédiat (pour les points isolés)
        cv2.circle(target_img, (x, y), LINE_THICKNESS // 2, 0, -1)
        dirty_canvas = True
        dirty_overlay = True

    # SOURIS BOUGE : DESSIN CONTINU
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and prev_pt is not None:
            # On dessine du point PRÉCÉDENT vers l'ACTUEL
            cv2.line(target_img, prev_pt, (x, y), color=0, 
                     thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)
            # CRUCIAL : Le point actuel devient le précédent pour la suite
            prev_pt = (x, y)
            dirty_canvas = True
            if current_mode == 1:
                dirty_overlay = True

    # CLIC RELÂCHÉ : FIN DU TRAIT
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        prev_pt = None  # On oublie le dernier point pour ne pas relier au prochain clic

def main():
    global current_mode, img_A, img_B, dirty_canvas, dirty_overlay, display_buffer
    
    # WINDOW_GUI_NORMAL aide parfois sur Mac pour éviter les soucis de DPI
    cv2.namedWindow('Dessin', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('Dessin', mouse_callback)
    
    # Pré-calcul du cache "Fantôme" de A
    ghost_A = img_A.copy()

    print("--- COMMANDES ---")
    print("[1] : Forme A")
    print("[2] : Forme B (Onion Skin)")
    print("[c] : Clear")
    print("[s] : Save")
    print("[q] : Quit")

    while True:
        
        # On ne recalcule l'affichage QUE si quelque chose a changé
        if dirty_canvas:
            
            if current_mode == 1:
                # Mode A simple
                final_view = img_A
                text = "Mode: A"
                col = (0, 0, 255)
            else:
                # Mode B avec Onion Skin
                # On ne met à jour le fantome que si A a changé
                if dirty_overlay:
                    ghost_A = cv2.max(img_A, 200) # Noir -> Gris
                    dirty_overlay = False
                
                # Fusion rapide
                final_view = cv2.min(ghost_A, img_B)
                text = "Mode: B"
                col = (0, 200, 0)

            # Conversion Couleur pour le texte (coûteux, donc fait uniquement si dirty)
            display_buffer = cv2.cvtColor(final_view, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_buffer, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, col, 2, cv2.LINE_AA)
            
            # Affichage
            cv2.imshow('Dessin', display_buffer)
            dirty_canvas = False

        # Sur MAC, waitKey(1) est parfois trop rapide ou erratique.
        # Si ça lag toujours, essaie waitKey(5) pour laisser respirer l'OS.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('s'):
            cv2.imwrite(FILENAME_A, img_A)
            cv2.imwrite(FILENAME_B, img_B)
            print("Saved!")
            # Petit feedback visuel (inversion couleurs)
            cv2.imshow('Dessin', 255 - display_buffer)
            cv2.waitKey(100)
            dirty_canvas = True # Force le rafraichissement normal
            break

        elif key == ord('1'):
            current_mode = 1
            dirty_canvas = True
        
        elif key == ord('2'):
            current_mode = 2
            dirty_canvas = True
            dirty_overlay = True # On doit recharger A
        
        elif key == ord('c'):
            if current_mode == 1:
                img_A[:] = 255
                dirty_overlay = True
            else:
                img_B[:] = 255
            dirty_canvas = True

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()