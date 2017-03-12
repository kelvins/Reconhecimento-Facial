# Algoritmos de Detecção e Reconhecimento Facial

Este repositório foi criado com a finalidade de compartilhar os algoritmos desenvolvidos durante o trabalho de mestrado intitulado "**Comparação de Técnicas de Reconhecimento Facial para Identificação de Presença em um Ambiente Real e Semicontrolado**", com o objetivo de facilitar e tornar possíveis futuras replicações dos experimentos realizados.

O trabalho desenvolvido faz parte da linha de pesquisa de "**Monitoramento de Presença**" do projeto "**Ensino e Monitoramento de Atividades Físicas via Técnicas de Inteligência Artificial**" (Processo 2014.1.923.86.4, publicado no DOE 125(45), em 10/03/2015), realizado conjuntamente pela **Universidade de São Paulo**, **Faculdade Campo Limpo Paulista** e **Academia Central Kungfu-Wushu**.

Basicamente foram utilizados os algoritmos:

 * **[Viola-Jones](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)**
 * **[Eigenfaces](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces)**
 * **[Fisherfaces](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#fisherfaces)**
 * **[Local Binary Patterns Histogram (LBPH)](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)**
 * **[Scale-Invariant Feature Transform (SIFT)](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html)**
 * **[Speeded Up Robust Features (SURF)](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html)**
 * **[Transformada Wavelet discreta](https://www.mathworks.com/help/wavelet/ref/dwt2.html)**

Trabalho desenvolvido por **Kelvin Salton do Prado** sob orientação do **Dr. Norton Trevisan Roman**.

**Nota**: a maior parte dos algoritmos desenvolvidos fazem uso da biblioteca **[OpenCV](http://opencv.org)**.

----------------------

### Open Source Computer Vision Library (OpenCV)

A **[OpenCV](http://opencv.org)** (Open Source Computer Vision Library) é uma biblioteca de visão computacional de código aberto e multiplataforma, originalmente desenvolvida pela Intel por volta do ano 2000, e livre tanto para uso acadêmico como para uso comercial. A **OpenCV** disponibiliza diversos algoritmos de visão computacional, como por exemplo filtros de imagem, reconhecimento de objetos e reconhecimento facial, e possui ainda suporte a várias linguagens, como Python, C++, Ruby, Matlab, entre outras.

O objetivo principal da biblioteca **OpenCV** é prover uma infraestrutura de visão computacional simples para auxiliar os desenvolvedores a construir aplicações sofisticadas de forma relativamente rápida.



### OpenCV Contrib

O **[OpenCV Contrib](https://github.com/opencv/opencv_contrib)** é um repositório criado para o desenvolvimento, armazenamento e compartilhamento de alguns módulos extras do **OpenCV**. Novos módulos que ainda não possuem uma interface de programação de aplicação (API) estável são disponibilizados neste repositório para, futuramente, quando o módulo evoluir e ganhar popularidade, ser movido para o repositório central do **OpenCV**, e então ser disponibilizado oficialmente com a biblioteca.

Cabe aqui ressaltar que os algoritmos de detecção e reconhecimento facial utilizados no trabalho foram fornecidos tanto pela biblioteca oficial da **OpenCV** quanto pelos módulos extras disponibilizados pelo **OpenCV Contrib**.

----------------------

Eventuais dúvidas sobre os algoritmos desenvolvidos podem ser esclarecidas através do e-mail **kelvinpfw@hotmail.com**
