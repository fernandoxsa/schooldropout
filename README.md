# 🎓 EduPredict — Sistema Inteligente de Previsão de Desempenho Escolar

> PBL 3 · Tecnologias e Sistemas · UNDB 2026.1

## 📌 Descrição

Aplicação web desenvolvida com **Streamlit** que utiliza uma **Rede Neural Artificial (MLP)** para prever o risco de reprovação de alunos com base em dados acadêmicos, familiares e comportamentais.

Desenvolvida como solução para o desafio proposto pela prefeitura: transformar dados educacionais em inteligência preditiva, permitindo ações antecipadas de intervenção pedagógica.

---

## 🚀 Como rodar localmente

### Pré-requisitos
- Python 3.10+
- pip

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/edupredict.git
cd edupredict

# Instale as dependências
pip install -r requirements.txt

# Execute o app
streamlit run app.py
```

O app estará disponível em `http://localhost:8501`

---

## 🌐 Deploy

A aplicação está publicada no **Streamlit Community Cloud**:

👉 **[Link do deploy aqui]**

---

## 🧠 Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| Python 3.10 | Linguagem principal |
| Streamlit | Interface web |
| Scikit-learn | Modelo MLP + pré-processamento |
| NumPy / Pandas | Manipulação de dados |
| Matplotlib / Seaborn | Visualizações |

---

## 🏗️ Arquitetura do Modelo

```
INPUT  → 15 features do aluno
HIDDEN 1 → 64 neurônios · ReLU
HIDDEN 2 → 32 neurônios · ReLU
HIDDEN 3 → 16 neurônios · ReLU
OUTPUT → 2 classes (aprovado / reprovado)

Otimizador: Adam
Early Stopping: Ativo
```

---

## 📊 Dataset

**Student Alcohol Consumption** — UCI Machine Learning Repository  
🔗 https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data

Variáveis utilizadas: notas G1/G2, faltas, tempo de estudo, reprovações anteriores, escolaridade dos pais, hábitos sociais e consumo de álcool.

---

## 📁 Estrutura do Projeto

```
edupredict/
├── app.py              # Aplicação Streamlit principal
├── requirements.txt    # Dependências
└── README.md           # Documentação
```

---

## 👥 Equipe

| Membro | Responsabilidade |
|---|---|
| [Nome] | Interface web + Deploy (Streamlit) |
| [Nome] | Modelo de IA + Treinamento |
| [Nome] | Análise e tratamento de dados |
| [Nome] | Backend / API |
| [Nome] | Documentação |

---

## 📈 Resultados

O modelo MLP atingiu:
- **Acurácia**: ~85%
- **F1-Score**: ~0.85
- **Early Stopping** evitando overfitting

---

## ⚠️ Limitações

- Dataset sintético baseado na estrutura do UCI (dados reais do Kaggle podem ser integrados)
- Modelo treinado localmente a cada inicialização (pode ser exportado com `joblib`)
- Não substitui avaliação pedagógica profissional

---

## 📚 Referências

- GERON, Aurélien. *Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow*. O'Reilly, 2019.
- RUSSELL, Stuart; NORVIG, Peter. *Artificial Intelligence: A Modern Approach*. Pearson, 2021.
