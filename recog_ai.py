import json
from langchain.chat_models  import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import markdown
import os


class recognition_assistant():
    def __init__(self, db):
        self.db = db
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    

    def getModuleSuggestions(self, doc):
        docs = self.db.similarity_search_with_score(doc, 5)
        
        module_suggestions = []
        for module, score in docs:
            module_info = {
                "title": module.metadata['title'],
                "credits": module.metadata['credits'],
                "workload": module.metadata['duration'],
                "description": module.metadata['description'],
                "level": module.metadata['level'],
                "program": module.metadata['program'],
                "content": module.page_content,
                "json": json.dumps(module.metadata),
            }
            module_suggestions.append(module_info)
        
        # Return the module suggestions
        return module_suggestions
    

    def getModulInfo(self, doc):
        systemmessage = """Analysiere die gegebenen Modulbeschreibungen und fülle anschließend die Lücken in folgendem JSON sinvoll aus. Der Workload sollte in Stunden pro Semester angegeben sein. Das Level bezieht sich auf das Bildungsniveau des Kurses und kann nur "Bachelor" oder "Master" enthalten. Wenn eine passende Information in der gegebenen Modulbeschreibung fehlt, soll das Attribut den Wert null bekommen. Achte darauf dass das Ergebnis valides JSON ist.
        {
            "title": "",
            "credits": "",
            "workload": "",
            "learninggoals": [],
            "assessmenttype": "",
            "level": ""
        }
        """

        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=self.openai_api_key, request_timeout=40, max_retries=2)

        messages = [
            SystemMessage(content=systemmessage),
            HumanMessage(content=doc)
        ]

        return chat(messages).content


    def getExaminationResult(self, module_internal, module_external):
        systemmessage = """
            Ich bin als Assistent*in im Prüfungsausschuss einer Hochschule tätig. Meine Hauptaufgaben umfassen die Beantwortung von Fragen zu Modulen und die Überprüfung, ob ein externes Modul auf ein internes Modul angerechnet werden kann.

            Folgende Kriterien werden bei der Prüfung der Anrechenbarkeit berücksichtigt:
            - Lernziele
            - ECTS-Punkte/Credits
            - Arbeitsaufwand
            - Bildungsniveau
            - Prüfungsform

            Bei der Bewertung werden diese Kriterien gleichwertig berücksichtigt.
            Eine Ausnahme ist der Arbeitsaufwand. Dieser sollte nicht in die Bewertung einfließen, wenn die Infromationen dazu nicht gut vergleichbar sind.
            Beide Module sollten möglichst demselben Bildungsniveau (Bachelor oder Master) entsprechen.
            Wenn das externe Modul mehr Credits aufweist als das interne Modul oder die Diskrepanz etwa 10 Prozent beträgt, ist dies kein Grund für eine Nichtanerkennung. Wenn das interne Modul jedoch signifikant mehr Credits hat als das externe Modul, kann höchstens eine teilweise Anerkennung erfolgen.
            Das Kriterium der Prüfungsform sollte nicht berücksichtigt werden wenn, diese Informatione nicht für beide Module vorliegt und nicht vergleichbar ist.

            Es gibt drei mögliche Ergebnisse für die Prüfung:
            - Vollständige Anrechnung, wenn mindestens 80 Prozent der Lernziele übereinstimmen
            - Teilweise Anrechnung, wenn mindestens 50 Prozent der Lernziele übereinstimmen
            - Keine Anrechnung, wenn nur wenige oder keine Lernziele übereinstimmen

            Die Abschnitte und Inhalte meiner Antworten strukturiere ich mit Markdown. Kriterien werden einzeln bewertet. Lernziele müssen nur bei Unterschieden aufgelistet werden. 
            Am Schluss der Prüfung folgt eine prägnante, hervorgehobene Zusammenfassung des Prüfungsergebnisses mit dem Ergebnis: *Vollständige Anrechnung*, *Teilweise Anrechnung* oder *Keine Anrechnung*.
        """

        humanmessage = """
            ## Externes Modul 

            """ + module_external + """


            ## Internes Modul:

            """ + module_internal + """
        """

        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=self.openai_api_key, request_timeout=60, max_retries=2)

        messages = [
            SystemMessage(content=systemmessage),
            HumanMessage(content=humanmessage)
        ]

        return markdown.markdown(chat(messages).content)