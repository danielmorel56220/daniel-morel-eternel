# Accès DME — liens utiles

Fiche simple pour retrouver les adresses du chatbot.

Base Railway :  
`https://daniel-morel-eternel-production.up.railway.app`

---

## 1. Chat tout public (clients / visiteurs)

**Adresse complète à partager :**  
https://daniel-morel-eternel-production.up.railway.app/chat-public

- Pas de mot de passe
- Interface « voix Daniel » pour le public
- C’est le lien à donner aux clients

---

## 2. Espace propriétaires (privé, avec mot de passe)

**Adresse complète :**  
https://daniel-morel-eternel-production.up.railway.app/admin

- Demande un **mot de passe** à l’écran
- Accès plus libre (prompt admin)
- **Ne pas partager** ce lien publiquement

### Où est le mot de passe ?

Il n’est **pas** écrit ici (sécurité / Git).

Il est stocké sous le nom : **`ADMIN_PASSWORD`**

| Endroit | Rôle |
|---|---|
| Railway → Variables | Mot de passe utilisé **en production** (celui du site en ligne) |
| Fichier `.env` à la racine du projet (local) | Pour tests sur ton Mac — à ajouter si absent |

Pour le retrouver ou le changer :  
Railway → projet DME → **Variables** → `ADMIN_PASSWORD`

---

## 3. Si le chat ne répond pas

Ordre de vérif rapide :

1. **Crédits Claude** → https://console.anthropic.com/settings/billing  
   (sans crédits, le chat public et admin restent en erreur)
2. **Supabase** actif (pas « en pause ») → dashboard projet `igdodyugqyeprtufohea`
3. **Santé API** → https://daniel-morel-eternel-production.up.railway.app/sante

## 4. Autres liens techniques

| Service | Lien |
|---|---|
| Santé API | https://daniel-morel-eternel-production.up.railway.app/sante |
| Supabase (base) | https://supabase.com/dashboard → projet `igdodyugqyeprtufohea` |
| Anthropic (crédits Claude) | https://console.anthropic.com/settings/billing |

---

*Dernière mise à jour : 13 août 2026*
