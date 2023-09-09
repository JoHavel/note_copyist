# Specifikace [ročníkového projektu](http://is.cuni.cz/studium/predmety/index.php?do=predmet&kod=NPRG045) <br> NoteCopyist

## Zadání
Cílem tohoto ročníkového projektu (NoteCopyist – Syntéza hudebních symbolů pomocí generativních neuronových sítí)
je připravit framework pro experimentování s generativními neuronovými sítěmi hlavně pro použití na hudebních symbolech.
Primárně je třeba naprogramovat konfigurovatelné generativní neuronové sítě jako modely v TensorFlow,
sekundárně je třeba připravit celý workflow pro práci s obrázky (a datasety obrázků) hudebních symbolů.

## Motivace
K rozpoznávání hudební notace máme relativně málo anotovaných trénovacích dat. Proto chceme tato data
generovat synteticky. K dispozici již máme [Mashcimu](https://github.com/Jirka-Mayer/BachelorThesis),
která umí z obrázků jednotlivých hudebních symbolů sesyntetizovat celý zápis. My se zaměříme právě
na ty obrázky jednotlivých symbolů, kterých je sice více než anotovaných celých zápisů, ale pořád relativně málo.
Tyto se budeme snažit syntetizovat pomocí generativních neuronových sítí. 

## Úvod do problematiky

### Generativní neuronové sítě
Neuronové sítě jsou již naprogramované v knihovně [TensorFlow](https://www.tensorflow.org/).
Naším cílem bude použít tuto knihovnu k naprogramování generativních neuronových sítí, jak jsou popsané v
[GAN](https://arxiv.org/abs/1406.2661), [AE](https://arxiv.org/abs/2003.05991) and [VAE](https://arxiv.org/abs/1312.6114), [AAE](https://arxiv.org/abs/1511.05644).

### Hudební notace
Ze znalostí hudební notace potřebujeme hlavně to, že většina symbolů má bod (námi nazvaný attachment point),
kde se váže k notové osnově. Jinak na obrázky hudebních symbolů pohlížíme jako na „jiné obrázky“.

### Operace s obrázky
Při úpravě obrázků budeme používat samozřejmě to, co máme k dispozici v samotném TensorFlow,
ale složitější operace s obrázky vezmeme z knihovny OpenCV (Open Source Computer Vision library).

## Hlavní funkce
- Na *jeden příkaz* natrénovat zvolenou generativní neuronovou síť se zvolenými parametry na zvolených trénovacích datech
  (dataset co nejvíce automaticky připravit, např. stáhnout, *vycentrovat na „attachment point“*, atd.).
- Při trénování zobrazit uživateli vzorky generovaných obrázků (pro analýzu procesu trénování a ukázku výsledku)
- S použitím natrénované sítě vygenerovat zvolený počet (případně počty potřebné pro Mashcimu) od každého symbolu.
- Umožnit přidat při generování binarizaci

## Struktura
- Zásadní část budou třídy implementující generativní neuronové sítě (`generators`)
  - Jednotlivé součásti generativních neuronových sítí (discriminator, decoder, encoder) lze použít samostatně, tedy je oddělíme do separátního modulu (`parts`)
- Potom potřebujeme zpracovávat datasety (`datasets`)
- Nějaké další podpůrné funkce pro vizualizaci a binarizaci (`utils`)
- A nakonec část spojující předchozí do jednoho pro jednoduché použití (`experiments`)
  - Pro tento účel navíc přidáme do předchozích částí „interface“, aby se všechny generativní neuronové sítě, datasety, atd. dali použít záměnitelně

Zvlášť pak budou části specifické pro hudební symboly, například vycentrování obrázků na attachment point, přidání attachment
points pro praporky, vygenerování obrázků přesně pro mashcimu.

## Detaily


