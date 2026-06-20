Note sul codice:

Ho deciso di riscrivere tutto da zero in quanto insoddisfatto da molte
librerie esistenti come “transformers” di huggingface, “xformers” o
anche l’implementazione di BLT pubblicata da Meta.

Ho l’impressione che si tratta di codebase sfuggiti un po’ di mano,
macchinosi, con codici che eseguono funzioni riassumibili in poche righe
che vengono estesi su migliaia di linee. Catene di if/then/else,
riassegnazioni di variabili nelle classi, funzioni aggiuntive utili solo
per casi molto specifici, determinati esperimenti o compatibilità legacy
con tante architetture diverse...

L’idea è di avere una libreria estremamente compatta, che chiunque possa
leggere in un pomeriggio ma capace comunque di far funzionare in modo
efficiente modelli come BLT, “bert-like”, “gpt-like” o Diffusion
Transformers. Vorrei riuscire ad ottenere classi il più possibile
astratte, facilmente combinabili tra loro ed evitare di avere codice
compatibile necessariamente con ogni architettura esistente.

Il codice che condivido è lontano dall’essere completo, però permette
già di addestrare un modello “di entropia” (primo step per ottenere un
BLT). È già a portata di mano scrivere le procedure per addestrare
modelli come bert o gpt, mentre sono vicinissimo a finire la parte su
BLT.

In generale, anche io sono andato un po’ fuori dalle specifiche che mi
ero proposto di eleganza e compattezza ma la situazione è ancora molto
recuperabile per esempio disegnando un buon dizionario “config” che i
vari componenti del modello si passano, per evitare l’esplosione di
variabili e creando uno script di training astratto che possa riciclare
i componenti necessari per l’addestramento dei modelli.

Cosa manca, principalmente:

- Una revisione del codice, che dovrebbe essere fatta da una persona
  diversa da me sia per esperienza sia per notare possibilissime sviste;
- Studiarsi un attimo come funzionano i nested/jagged tensors in
  PyTorch. Potrebbe darsi che migrare verso questo tipo di tensori
  riesca a farci giungere a un risultato simile all’unpadding. Io non ho
  fatto a tempo, ma può svoltare il lavoro e permetterci finalmente di
  avere il codice necessario per addestrare Albertone/Ina;
- Rivedere l’efficienza della FlexAttention: al momento le blockmask
  vengono calcolate in alcuni casi nell’init, in altri nella forward.
  C’è da vedere se in alcuni casi questo può portare a dei colli di
  bottiglia. In generale, trattandosi di un punto molto delicato, è
  necessario guardarlo con cura e valutare come ottimizzare le
  performance ad esempio con la compilazione;
- Scrivere una bella procedura astratta di training, del tipo file
  config.json che definisce modello, file train_config.json con la
  procedura di training, path dell’atlas dataset di addestramento e
  test, possibilità di interrompere l’addestramento e riprendere da
  checkpoint, logging su TensorBoard o W&B. Facile farlo funzionare, più
  difficile ma soddisfacente massimizzare l’eleganza in modo da poterlo
  riciclare per scopi futuri;
- Integrazione con HuggingFace; non dovrebbe essere troppo complicato,
  ma sicuramente è noioso; l’ideale è avere un file modeling_xxx.py che
  importi la libreria e funga da interfaccia con i vari componenti HF.

Poi, qualora ci fosse interesse nel lavorare sull’architettura
sperimentale, bisogna ancora fare la parte di Diffusion Transformer
(facile) e capire come definire il router, la multi-loss e il batch
multimodale (più delicato).

In generale, la semplicità del codice è ispirata a <a href="https://github.com/Emericen/tiny-qwen">TinyQwen</a>, un piccolo progetto che permette di far girare Qwen in poche righe di Pytorch.
