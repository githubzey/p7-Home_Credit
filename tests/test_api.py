

import unittest
from fastapi import status
import requests
import json
class TestTools(unittest.TestCase):
    """Test lower round."""

    def setup(self):
        self.url = "https://apihomecredit-861d00eaed91.herokuapp.com/predict" 
        self.client_data_json_new ="[{\"EXT_SOURCE_2\":0.0304315707,\"EXT_SOURCE_3\":0.1107962886,\"CODE_GENDER\":0.7209239877,\"PAYMENT_RATE\":-0.2418253653,\"DAYS_EMPLOYED\":0.2833922874,\"PREV_CNT_PAYMENT_MEAN\":-0.2903476431,\"NAME_EDUCATION_TYPE_Highereducation\":1.7625702765,\"INSTAL_DPD_MEAN\":-0.052079095,\"DAYS_BIRTH\":-0.9914858035,\"AMT_ANNUITY\":0.6444642056}]"

    def test_round_down(self):
        """
        Test lower round.

        :return:
        """
        url = "https://apihomecredit-861d00eaed91.herokuapp.com/predict" 
        client_data_json_new ="[{\"EXT_SOURCE_2\":0.0304315707,\"EXT_SOURCE_3\":0.1107962886,\"CODE_GENDER\":0.7209239877,\"PAYMENT_RATE\":-0.2418253653,\"DAYS_EMPLOYED\":0.2833922874,\"PREV_CNT_PAYMENT_MEAN\":-0.2903476431,\"NAME_EDUCATION_TYPE_Highereducation\":1.7625702765,\"INSTAL_DPD_MEAN\":-0.052079095,\"DAYS_BIRTH\":-0.9914858035,\"AMT_ANNUITY\":0.6444642056}]"
        response = requests.post(url, json={"data": client_data_json_new })
        prediction = response.json()["proba"]
        expected_result = 0.38
        assert response.status_code == 200
        assert prediction == expected_result
        