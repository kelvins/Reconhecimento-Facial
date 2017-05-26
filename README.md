# Trabalho

Este repositório foi criado com a finalidade de compartilhar os algoritmos desenvolvidos durante o trabalho de mestrado intitulado "**Comparação de Técnicas de Reconhecimento Facial para Identificação de Presença em um Ambiente Real e Semicontrolado**", com o objetivo de facilitar e tornar possíveis futuras replicações dos experimentos realizados.

O trabalho desenvolvido faz parte da linha de pesquisa de "**Monitoramento de Presença**" do projeto "**Ensino e Monitoramento de Atividades Físicas via Técnicas de Inteligência Artificial**" (Processo 2014.1.923.86.4, publicado no DOE 125(45), em 10/03/2015), realizado conjuntamente pela **Universidade de São Paulo**, **Faculdade Campo Limpo Paulista** e **Academia Central Kungfu-Wushu**.

Basicamente foram utilizados os algoritmos:

 * **[Viola-Jones](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)**
   * Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
 * **[Eigenfaces](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces)**
   * Turk, Matthew, and Alex Pentland. "Eigenfaces for recognition." Journal of cognitive neuroscience 3.1 (1991): 71-86.
 * **[Fisherfaces](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#fisherfaces)**
   * Belhumeur, Peter N., João P. Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.
 * **[Local Binary Patterns Histogram (LBPH)](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)**
   * Ojala, Timo, Matti Pietikäinen, and David Harwood. "A comparative study of texture measures with classification based on featured distributions." Pattern recognition 29.1 (1996): 51-59.
 * **[Scale-Invariant Feature Transform (SIFT)](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html)**
   * Lowe, David G. "Distinctive image features from scale-invariant keypoints." International journal of computer vision 60.2 (2004): 91-110.
 * **[Speeded Up Robust Features (SURF)](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html)**
   * Bay, Herbert, Tinne Tuytelaars, and Luc Van Gool. "Surf: Speeded up robust features." European conference on computer vision. Springer Berlin Heidelberg, 2006.
 * **[Transformada Wavelet discreta](https://www.mathworks.com/help/wavelet/ref/dwt2.html)**
   * Burrus, C. Sidney, Ramesh A. Gopinath, and Haitao Guo. "Introduction to wavelets and wavelet transforms: a primer." (1997).

Trabalho desenvolvido por **Kelvin Salton do Prado** sob orientação do professor **Dr. Norton Trevisan Roman**.

**Nota**: a maior parte dos algoritmos desenvolvidos fazem uso da biblioteca **[OpenCV](http://opencv.org)**. Instruções sobre a instalação da **OpenCV** no **Ubuntu** podem ser encontradas **[aqui](https://github.com/kelvins/Reconhecimento-Facial/wiki/Instalando-OpenCV-no-Ubuntu)**.

----------------------

### Open Source Computer Vision Library (OpenCV)

A **[OpenCV](http://opencv.org)** (Open Source Computer Vision Library) é uma biblioteca de visão computacional de código aberto e multiplataforma, originalmente desenvolvida pela Intel por volta do ano 2000, e livre tanto para uso acadêmico como para uso comercial. A **OpenCV** disponibiliza diversos algoritmos de visão computacional, como por exemplo filtros de imagem, reconhecimento de objetos e reconhecimento facial, e possui ainda suporte a várias linguagens, como Python, C++, Ruby, Matlab, entre outras.

O objetivo principal da biblioteca **OpenCV** é prover uma infraestrutura de visão computacional simples para auxiliar os desenvolvedores a construir aplicações sofisticadas de forma relativamente rápida.



### OpenCV Contrib

O **[OpenCV Contrib](https://github.com/opencv/opencv_contrib)** é um repositório criado para o desenvolvimento, armazenamento e compartilhamento de alguns módulos extras do **OpenCV**. Novos módulos que ainda não possuem uma interface de programação de aplicação (API) estável são disponibilizados neste repositório para, futuramente, quando o módulo evoluir e ganhar popularidade, ser movido para o repositório central do **OpenCV**, e então ser disponibilizado oficialmente com a biblioteca.

Cabe aqui ressaltar que os algoritmos de detecção e reconhecimento facial utilizados no trabalho foram fornecidos tanto pela biblioteca oficial da **OpenCV** quanto pelos módulos extras disponibilizados pelo **OpenCV Contrib**.

----------------------

### Documentação

A documentação do projeto FaceRecognition pode ser acessada aqui: [![FaceRecognition Docs](https://img.shields.io/badge/documentation-FaceRecognition-blue.svg)](https://kelvins.github.io/Reconhecimento-Facial/build/html/index.html)

Outros documentos relacionados aos demais códigos e ao repositório em geral podem ser encontrados na **[Wiki](../../wiki)**.

----------------------

### Dúvidas

Eventuais dúvidas sobre os algoritmos ou sobre a pesquisa podem ser esclarecidas através do e-mail **kelvinpfw@hotmail.com**
