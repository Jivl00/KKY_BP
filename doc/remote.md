# SSH připojení (Putty)
- autentizace heslem nebo klíčem – vygenerování PuTTygen, Pageant
- Putty – vytvoření nebo načtení již vytvořené session
- vytvoření tunelu – SSH – Tunnels – vyplnění source port a destination
- spuštění Jupyteru:
```
jupyter lab --no-browser --port=8080
```
# Přenos souborů pomocí WinSCP
Jako Total Commander, slouží k přenosu souborů mezi lokálním a vzdáleným počítačem
# Základní příkazy na Linuxu
- změna adresáře – cd
- vypsání adresáře – ls
- kopírování souborů/složek – cp (cp -r)
- přesunutí souborů/složek – mv <source> <destination>
- přejmenování souborů/složek – mv
- vytvoření souborů/složek – touch empty_file.txt, mkdir
- smazání souborů/složek – rm (rm -r)
- editace textových souborů pomocí nano
# Práce s Tmux
- vytvoření session – tmux new, tmux new -s mysession
- zrušení session – tmux kill-session -t mysession
- připojení k session – tmux attach -t mysession
- odpojení od session – ctrl + b d
---
