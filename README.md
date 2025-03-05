# RagApp
## Introductie
### De opdracht
Ontwikkel een Streamlit-applicatie waarin een PDF-document (specifiek een jaarverslag van KPN) in een vector database wordt geplaatst en een chatfunctionaliteit wordt toegevoegd waarmee gebruikers vragen kunnen stellen over de inhoud van het document.

### De toepassing
Bij het ontwikkelen van een RAG applicatie zijn er een aantal overwegingen die gemaakt moeten worden: Welke vector ga ik gebruiken? Wat is de beste chunking methode voor deze toepassing? Hoe kan ik deze chunks het best embedden? Welke bestaande frameworks kan ik gebruiken om het proces te versnellen? De antwoorden hierop, samen met mijn onderbouwing, zijn hieronder terug te vinden.

## Onderbouwing 
### Vector database
Aangezien het een vrij eenvoudige toepassing is, waarbij er maar 1 relatief klein pdf-document moet worden toegevoegd, is de keuze gevallen op een open-source oplossing. De meeste closed-source modellen, zijn veel complexer en hebben veel meer opties. Dit zorgt er dus ook voor dat we een vector database kunnen gebruiken die gewoon lokaal draait in plaats van ergens op een server (Pinecone). 

De open-source opties die ik heb bekeken waren ChromaDB, FAISS, Qdrant en Weaviate. Hierbij is snelheid van ontwikkeling weer de belangrijkste factor geweest in mijn keuze. FAISS viel al snel af omdat dit geschikter was voor toepassingen op hele grote schaal. Qdrant en Weaviate bieden geavanceerdere opties, wat er dus ook voor zorgt dat de leercurve steiler is. ChromaDB is een redelijke 'lightweight' oplossing die de features bevat die ik nodig had om deze toepassing snel te ontwikkelen. Daarom heb ik dus gekozen voor **ChromaDB**.

### Chunking strategie
Voordat ik inga op de chunking strategie wil ik eerst mijn methode bespreken om content uit de pdf te halen. Ik heb recent veel gelezen over Pymupdf4LLM, een library die content uit een pdf converteert naar markdown. Markdown, zoals er in veel blogs wordt besproken, zou beter zijn voor RAG applicaties omdat het de LLM meer context over de soort tekst biedt (headers, dikgedrukte tekst, etc.). Dit biedt daarnaast de mogelijkheid om te chunken door gebruik te maken van deze markdown context, maar daarover later meer. Ik heb ook overwogen om afbeeldingen te embedden. Dit zou een complexer proces zijn, waarbij ik van plan was om een LLM eerst een gedetailleerde beschrijving van de afbeelding te laten genereren. Deze zou ik vervolgens koppelen aan de afbeelding zelf, en tijdens het retrieval proces, als deze afbeelding beschrijving opgehaald wordt, zou ik behalve de tekstuele omschrijving, ook de afbeelding zelf als input aan het model geven om zo de vraag beter te beantwoorden.

Nu de chunking strategie. Omdat ik de tekst als markdown had dacht ik eraan om ```MarkdownHeaderTextSplitter``` van Langfuse te gebruiken. Uiteindelijk bleek dit dus toch wat lastiger, en met het oog op snel ontwikkelen heb ik dus gekozen voor de ```MarkdownTextSplitter```, ook van Langfuse. Dit is een simpele text splitter die is gebaseerd op de ```RecursiveCharacterTextSplitter```, hier hebben we dus alleen een chunk size en een overlap. De markdown splitter is anders dan de recursive splitter in het feit dat deze de markdown opmaak behoudt, die we dus nodig hebben om meer context aan het model te geven.

Ik heb gekozen voor een chunk size van 1000 karakters. Ik heb ook wat korte testjes uitgevoerd met chunk sizes van 400, 800 en 1500, maar 1000 was de sweet spot tussen groot genoeg om belangrijke data die bij elkaar hoort te behouden, maar niet te groot dat het resulteert in meer noise en hogere kosten door het grotere aantal tokens. De overlap van 100 zorgt dat er genoeg context overblijft tussen de chunks.

De ```MarkdownHeaderTextSplitter``` was mooi geweest omdat er dan gesplit zou worden op headers. Ervan uitgaande dat de Pymupdf4LLM goed zou werken, dan zou alle semantische data binnen een stuk tekst (afgebakend door een header) binnen 1 chunk kunnen blijven. Maar door teveel implementatie tijd, en een Pymupdf4LLM toepassing die niet optimaal werkt, zorgde dit ervoor dat ik een hele hoop chunks kreeg. Veel headers worden nog niet juist geformat. Een mogelijke fix hiervoor zou kunnen zijn om de pdf content door gemini te laten converteren naar markdown. Ik denk dat LLM toepassingen hier beter in zijn en de nieuwe Gemini Flash is super snel en heeft een enorme context window.

### Embedding strategie
Over de embedding strategie kan ik heel kort zijn. Ik heb gekozen voor een embedder van openAI. Dit was voornamelijk omdat ChromaDB een geïntegreerde openAI embeddingsfunctie heeft waarmee je met 2 regels code een embedder op kan zetten. Ik heb gekozen voor de ```text-embedding-3-small```, uit mijn onderzoekje kwam naar voren dat deze goedkoper is dan ```text-embedding-ada-002``` en vergelijkbaar, en op sommige benchmarks zelfs beter (medium artikel [hier]([url](https://medium.com/@arundona/open-ai-3rd-gen-embedding-models-whats-driving-the-improvements-4c23b88751f1))).

Performance zou nog verbeterd kunnen worden door ```text-embedding-3-large``` te gebruiken maar dat vond ik voor deze toepassing niet nodig.

### Verbetermogelijkheden
Ik heb in bovenstaande tekst al wat verbetermogelijkheden genoemd, zoals: 
* Techniek om afbeeldingen te embedden
* LLM (Gemini) gebruiken om pdf naar markdown te converteren
* De ```MarkdownHeaderTextSplitter``` toepassen op goede markdown om zo semantisch vergelijkbare data binnen 1 chunk te houden
* Groter/beter embedding model

Een andere techniek zou kunnen zijn om multi-stage retrieval toe te passen waarbij je dus eerst breed zoekt naar relevante secties, waarna je in de volgende zoekfase binnen de geselecteerde relevante secties gaat zoeken, daarna voeg je alles samen tot 1 resultaat wat je in generatie fase weer aan het model geeft.

Nu wil ik graag andere mogelijke verbeteringen bespreken. Een van deze verbeteringen is niet direct een RAG toepassing, maar meer LLM-gebaseerd. Omdat er nu (vooral) door Gemini stappen worden gezet met een grotere context window, in combinatie met efficiëntere modellen zou het mooi zijn om het hele RAG proces te vermijden en direct alle data in Gemini te voeren. Met een context van 1 miljoen tokens wordt je dus niet meer gelimiteerd door retrieval technieken, chunken of het embedden van tekst. Laat gewoon een LLM los op de tekst waarbij je dus ook niet meer wordt gelimiteerd doordat langeafstandsafhankelijkheden?(long distance dependencies) die verdwijnen door chunken, omdat je toch de hele tekst hebt.


## Mijn code
### Introductie
Ik heb een Streamlit-applicatie ontwikkeld die gebruikers in staat stelt om vragen te stellen over PDF-documenten, met name bedrijfsjaarverslagen. De gebruiker kan een PDF uploaden die vervolgens wordt verwerkt, in een vector database wordt opgeslagen en waarover vragen kunnen worden gesteld via een chat-interface.

### Functionaliteiten

#### Basis RAG-functionaliteit
De kerntaak van deze applicatie is het omzetten van PDF-documenten naar doorzoekbare vectoren. Hiervoor gebruik ik een combinatie van PDF-extractie, chunking, embedding en retrieval technieken. Wanneer een gebruiker een vraag stelt, zoekt het systeem de meest relevante fragmenten op en formuleert een contextueel antwoord.

#### Conversatiegeschiedenis
Een van de extra functionaliteiten die ik heb toegevoegd is de implementatie van een volledige conversatiegeschiedenis. Ik heb een `ConversationHandler`-klasse ontwikkeld die bijhoudt welke vragen en antwoorden al zijn uitgewisseld. Het systeem kan hierdoor automatisch detecteren wanneer een nieuwe vraag voortbouwt op eerdere vragen, wat resulteert in een natuurlijkere gebruikerservaring. 

Gebruikers hoeven niet steeds de volledige context te herhalen, wat vooral handig is bij het analyseren van complexe documenten zoals jaarverslagen. De detectie werkt door te kijken naar verwijswoorden, verbindingswoorden en de lengte van de vraag.

Ook is de 'geretrievde' data toegevoegd, maar dan collapsible zodat het gesprek mooi leesbaar blijft.

#### Verschillende perspectieven
Als bonus heb ik verschillende "rollen" geïmplementeerd waaruit de gebruiker kan kiezen. Hiermee kun je het document vanuit verschillende invalshoeken laten analyseren:

- Een standaard assistent voor algemene vragen
- Een bedrijfsjurist voor juridische implicaties
- Een econoom voor financiële analyses
- Een kritische journalist die inconsistenties onderzoekt
- Een theoloog voor ethische overwegingen

# RagApp gebruik

Een gedockeriseerde Retrieval-Augmented Generation (RAG) applicatie voor PDF-documentanalyse en vraag-antwoordfunctionaliteit.

## Functionaliteiten

- Uploaden en verwerken van PDF-documenten
- Vragen stellen over de inhoud van documenten
- AI-gestuurde antwoorden krijgen met bronvermeldingen
- Selecteren van verschillende perspectieven (Standaard, Bedrijfsjurist, Econoom, etc.)
- Gespreksgeschiedenis met detectie van vervolgvragen
- Responsieve gebruikersinterface gebouwd met Streamlit

## Vereisten

- Docker en Docker Compose
- OpenAI API-sleutel

## Snel aan de slag

### 1. Clone de repository

```bash
git clone https://github.com/derkdoel/RagApp.git
cd RagApp
```

### 2. Maak een `.env` bestand aan

Maak een `.env` bestand aan in de hoofdmap met je OpenAI API-sleutel:

```
OPENAI_API_KEY=jouw-openai-api-sleutel
```

### 3. Bouwen en uitvoeren met Docker Compose

```bash
docker-compose up --build
```

De applicatie is beschikbaar op http://localhost:8501

### 4. De applicatie gebruiken

1. Open de applicatie op http://localhost:8501
2. Gebruik de zijbalk om een PDF-document te uploaden
3. Klik op "Process PDF" om het document te analyseren
4. Stel vragen over het document in de chatinterface
5. Bekijk antwoorden en hun bronnen uit het document

## Configuratie

### Het OpenAI-model wijzigen

De applicatie gebruikt standaard `gpt-4o-mini`. Indien nodig kun je het model wijzigen in `src/chat/openai_client.py`.

### Permanente opslag

De vectordatabase wordt opgeslagen in de map `streamlit_chroma_db`, die als volume wordt gekoppeld in de Docker-container voor behoud tussen herstarts.

## Geavanceerde functies

### Antwoordperspectieven

Kies uit verschillende professionele perspectieven om antwoorden te krijgen die zijn afgestemd op specifieke gezichtspunten:

- Standaard Assistent
- Bedrijfsjurist
- Econoom
- Kritische Journalist
- Theoloog

### Gespreksbeheer

- De app houdt de gespreksgeschiedenis bij
- Detecteert automatisch vervolgvragen
- Optie om gesprekken te wissen of op te slaan

## Architectuur

Deze applicatie is gebouwd met:

- **Streamlit**: Webinterface
- **LangChain**: Documentverwerking en vectorzoekopdrachten
- **ChromaDB**: Vectordatabase
- **OpenAI**: Embeddings en tekstgeneratie
- **PyMuPDF4LLM**: PDF-verwerking
- **Docker**: Containerisatie

## Probleemoplossing

### Veelvoorkomende problemen

- **PDF-verwerking mislukt**: Zorg ervoor dat je PDF niet beveiligd is met een wachtwoord en in een standaardformaat is
- **API-verbindingsproblemen**: Controleer of je OpenAI API-sleutel correct is en voldoende quota heeft
- **Docker Volume-problemen**: Als de database niet behouden blijft, controleer dan de Docker-volume-machtigingen

## Ontwikkeling

Om de applicatie in ontwikkelingsmodus uit te voeren zonder Docker:

1. Installeer Poetry: `pip install poetry==1.7.1`
2. Installeer afhankelijkheden: `poetry install`
3. Start de app: `poetry run streamlit run src/streamlit_app.py`
