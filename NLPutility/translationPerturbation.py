##### using translation to generate paraphrased sentences #####
from google.cloud import translate

class translationPerturbation:
    def __init__(self, authFilePath='key/Paraphrasing-684a368e96ad.json'):
        '''
            Please provide your own google translation API key here 
        '''
        self.translate_client = translate.Client.from_service_account_json(authFilePath)

    def perturbSentence(self, inputSentence):
        print "\n\ntranslation perturbation:", inputSentence
        # The text to translate

        targLangs = [
            u'ar',
            u'zh',
            u'cs',
            u'da',
            u'nl',
            u'fr',
            u'de',
            u'iw',
            u'hi',
            u'it',
            u'ja',
            u'ko',
            u'tr'
        ]
        sentenceList = set()

        ### the default model is NMT
        for target in targLangs:
            # Translates some text into Russian
            translation = self.translate_client.translate(
                inputSentence,
                target_language=target)
            # print translation
            outputSentence = self.translate_client.translate(
                translation["translatedText"],
                source_language=target,
                target_language=u'en')
            # print outputSentence
            sentenceList.add(outputSentence["translatedText"])
        perturbedSen = list(sentenceList)
        print perturbedSen, "\n\n"
        return perturbedSen
