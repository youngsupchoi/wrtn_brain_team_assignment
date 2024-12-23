from os import system
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate

class Prompt():
  def __init__(self) -> None:
    pass  

  def get_evaluate_truthfulness_prompt(self) -> ChatPromptTemplate:
    system_message_prompt = """
      당신은 형법에 대해 잘 알고있는 변호사입니다. 주어진 요구사항을 엄격한 법적 사고로 판단해야합니다.
      
      주어진 문제와 선지는 대한민국 형법 문제의 하나의 선지입니다. 문제, 그리고 주어진 reference를 참고하여 선지가 옳은지 판단해주세요.
      자세한 정답 추론/해설 과정을 한글로 생성해야합니다.
      주어진 reference를 차근차근 나눠서 살펴보면서 문제 혹은 선지와 관련있는 것들이 어떤게 있는지 판단하고 이를 참고하여 선지가 옳은지 판단해주세요.
      만약 관련된 reference가 없다고 판단된다면 이를 명시하고 선지가 옳은지 판단해주세요.

      맞다면 True를 틀리다면 False를 선택해주세요. 만약 어느쪽에도 해당하지 않는다고 판단되면 False을 선택해주세요.

      ===========
      예시:
      law_reference:
      형법의 적용범위, 제1조(범죄의 성립과 처벌)
      ① 범죄의 성립과 처벌은 행위 시의 법률에 따른다.
      ② 범죄 후 법률이 변경되어 그 행위가 범죄를 구성하지 아니하게 되거나 형이 구법(舊法)보다 가벼워진 경우에는 신법(新法)에 따른다.
      ③ 재판이 확정된 후 법률이 변경되어 그 행위가 범죄를 구성하지 아니하게 된 경우에는 형의 집행을 면제한다.

      web_reference:
      공소시효는 개별 구성요건이 규정하고 있는 법정형을 기준으로, 최장 25년, 최단 1년이 경과하면 완성되는 것이 원칙이다(형사소송법 제249조 제1항).  공소제기 후, 판결의 확정 없이, 25년이 경과하면, 공소시효가 완성된 것으로 간주하는데 이를 의제공소시효(재판시효)라고 한다(형사소송법 제249조 제2항). 피고인의 소재불명으로 인한 영구미제사건을 종결처리하기 위한 규정이다.


      문제: 공소시효에 대한 설명으로 옳지 않은 것은?

      판단해야할 선지: 범죄 후 법률의 개정에 의하여 법정형이 가 벼워진 경우에는 형법 제1조 제2항에 의하여 당해 범죄 사실에 적용될 가벼운 법정형인 신법의 법정형이 공소시효 기간의 기준이 된다.

      review_law_reference:
      주어진 law_reference에 따르면, 형법 제1조 제2항은 다음과 같은 내용을 담고 있습니다:
      형법 제1조(범죄의 성립과 처벌)
      ② 범죄 후 법률이 변경되어 그 행위가 범죄를 구성하지 아니하게 되거나 형이 구법보다 가벼워진 경우에는 신법에 따른다.
      이 조항은 형벌불소급의 예외로, 범죄 후 법률이 변경되어 형이 가벼워진 경우에는 신법을 적용하여 처벌한다는 내용을 규정하고 있습니다. 이는 처벌에 있어서 피고인에게 유리한 법이 있을 경우 이를 적용하는 원칙을 나타냅니다.
      그러나, 공소시효 기간의 산정 기준에 대해서는 형법 제1조 제2항에서 언급하고 있지 않습니다. 공소시효의 기간 산정은 범죄 행위 시의 법정형에 따라 결정된다는 것이 일반적인 해석입니다.
      따라서, 주어진 reference에는 공소시효 기간 산정 기준이 신법의 법정형으로 변경될 수 있다는 내용은 포함되어 있지 않습니다.

      review_web_reference:
      주어진 reference에 따르면, 공소시효는 개별 구성요건에 규정된 법정형을 기준으로 최장 25년, 최단 1년의 기간이 설정되며 이는 형사소송법 제249조 제1항에 명시되어 있습니다. 또한, 공소제기 후 25년이 경과하면 판결 확정 없이 공소시효가 완성된 것으로 간주하는 의제공소시효(재판시효)가 있으며, 이는 형사소송법 제249조 제2항에서 규정하고 있습니다. 해당 조항은 공소시효의 기간 산정 기준이 법정형에 따라 결정된다는 점을 강조하며, 신법의 법정형으로 공소시효 기간이 변경된다는 내용은 포함되어 있지 않습니다. 따라서 선지의 판단과 관련된 내용은 reference에 포함되어 있지 않습니다.

      thinking_process:
      형법 제1조 제2항은 처벌에 관한 규정으로, 법정형이 가벼워진 경우 신법을 적용한다는 원칙을 명시하고 있습니다.
      그러나, 해당 조항은 공소시효 기간의 기준에 대해 명시적으로 다루지 않습니다. 공소시효는 범죄의 법정형에 따라 기간이 달라지며, 이는 범죄 행위 당시의 법정형을 기준으로 산정됩니다. 형벌불소급의 예외 규정이 공소시효 기간에까지 영향을 미친다는 내용은 관련 reference에 없습니다.
      주어진 law_reference에는 공소시효와 관련된 직접적인 내용은 없으며, 형법 제1조 제2항의 적용 범위는 처벌에 한정됩니다. 따라서, 선지에서 주장하는 공소시효 기간 산정 기준의 변경은 주어진 reference와 부합하지 않습니다.

      final_answer:
      False
      ===========

      law_reference:
      {law_reference}

      web_reference:
      {web_reference}

      문제: {question}
      판단해야할 선지: {choice}
      """
    
    system_prompt = SystemMessagePromptTemplate.from_template(
        template=system_message_prompt
    )
    return ChatPromptTemplate.from_messages([
      system_prompt,
    ])
  
      

  def get_find_target_prompt(self) -> ChatPromptTemplate:
    system_message_prompt = """
      주어진 문제의 의도를 파악해야합니다. 문제가 옳은 것을 고르는 것이라면 True, 옳지 않은 것을 False를 선택해주세요.
      주어진 문제가 선택 문항은 주어지지 않았습니다. 문제만 보고 의도를 파악해주세요.
      만약 어느쪽에도 해당하지 않는다고 판단되면 None을 선택해주세요.

      ===========
      예시:
      문제:
      공판기일의 절차에 대한 설명으로 옳지 않은 것은?

      thinking_process:
     공판기일의 절차에 대한 설명으로 옳지 않은 것을 고르라고 하고있기에 결과는 False가 됩니다. 

      target: False
      ===========
      문제:
      {question}
      """
    system_prompt = SystemMessagePromptTemplate.from_template(
        template=system_message_prompt
    )
    return ChatPromptTemplate.from_messages([
      system_prompt,
      ])
  def get_compete_choices_prompt(self) -> ChatPromptTemplate:
    system_message_prompt = """
      다음은 일반인이 형법에 대한 객관식 문제를 푼 과정입니다. 풀이중 합리적인 풀이는 단 한가지입니다.
      당신은 전문 변호사로서 아래 과정중 법적 사고에 부합하지 않거나 알고있는 법 지식에 맞지 않는 것들을 판단해야하고 이에 대해 설명해야합니다.
      그래서 최종적으로 맞는 답의 번호를 반환해야합니다.

      번호가 누락된 문제가 있을시 없다고 판단하고 풀이하면 됩니다.

      ===========
      예시:

      문제: 사기죄에 대한 설명으로 옳은 것은?
      선택지 및 일반인들의 풀이과정:
      1. 甲이 피해자 A에게 자동차를 매도하겠다고 거짓말하고 자동차를 양도하면서 소유권 이전등록에 필요한 일체의 서류를 교부하여 매매 대금을 수령한 다음, 자동차에 미리 부착해 놓은 지피에스 (GPS)로 위치를 추적하여 자동차를 가져간 경우, 甲에게 사기죄가 성립한다.
      사기죄는 기망행위에 의해 타인의 재물을 교부받는 행위로 성립합니다. 여기서 甲은 피해자 A를 기망하여 자동차를 받았고, 이후 GPS로 자동차를 가져간 경우, 명백히 사기죄가 성립합니다.

      2. 甲이 A에게 사업자 등록 명의를 빌려주면 세금이나 채무는 모두 자신이 변제하겠다고 속여 그로부터 명의를 대여받아 호텔을 운영한 경우, A가 명의를 대여하였다는 것만으로 사기죄의 처분행위가 있었다고 보기는 어렵다.
      여기서 A가 명의를 대여한 행위가 사기죄의 처분행위에 해당하지 않는다면, 사기죄가 성립하지 않을 수 있습니다. 그러나 A가 실제로 속임수에 의해 처분한 것이라면 사기죄가 성립할 수 있습니다.

      3. 甲이 피해자 A로 하여금 A의 예금을 인출하게 하고, 그 인출한 현금을 A의 집에 보관하도록 거짓말을 한 경우, A의 처분행위가 인정되어 甲에게 사기죄가 성립한다.
      A가 자신의 예금을 인출하게 하고 집에 보관하도록 속인 경우, A는 자신의 재산을 처분한 것으로 사기죄가 성립합니다.


      차근 차근 생각해보겠습니다
      thinking_process:
      1. 사기죄의 성립 요건은 기망행위 → 착오 → 처분행위 → 재산적 이익 취득입니다. 甲은 피해자 A를 기망하여 매매 대금을 수령하였으므로 기망행위와 처분행위가 성립한다고 볼 여지가 있지만 문제는 이후 甲이 GPS를 통해 다시 자동차를 가져간 행위인데, 이는 별도의 절도죄로 평가될 가능성이 있다는 부분입니다. 따라서 甲의 행위는 사기죄와 절도죄가 경합적으로 성립할 수 있으나, 사기죄만으로 규정하는 것은 법적 판단에 부합하지 않습니다.

      2. 사기죄가 성립하려면 처분행위가 있어야 하고, 이는 재산적 이익을 취득할 수 있는 행위여야 합니다. A가 단순히 명의를 대여한 것이라면, 이는 처분행위로 볼 수 없으며, 처분행위가 없으면 사기죄는 성립하지 않습니다. 여기서 "처분행위가 있었다고 보기는 어렵다"는 내용은 법적 판단에 부합합니다.

      3. 사기죄에서 처분행위란 피해자가 자신의 재산을 자발적으로 처분한 경우를 의미합니다. A는 자신의 예금을 인출했으나, 인출한 현금을 A의 집에 보관하도록 속임을 당했을 뿐, 甲이 재산적 이익을 취득했다고 보기 어렵습니다. 인출 및 보관 행위 자체는 처분행위로 인정되지 않으므로 사기죄 성립 요건에 부합하지 않습니다.

      final_answer:
      2

      ===========
      문제:
      {question}

      선택지 및 일반인들의 풀이과정:
      {merged_thinking_process_and_choice}

      차근 차근 생각해보겠습니다
      """

    system_prompt = SystemMessagePromptTemplate.from_template(
      template=system_message_prompt
    )
    return ChatPromptTemplate.from_messages([
      system_prompt,
    ])

  def get_criminal_law_qa_prompt(self) -> ChatPromptTemplate:
    system_prompt = self.__get_system_prompt()
    ai_prompt = self.__get_ai_prompt_template()

    return ChatPromptTemplate.from_messages([
      system_prompt, 
      ai_prompt
      ])

  def __get_system_prompt(self) -> SystemMessagePromptTemplate:
    system_message_prompt = """
      다음은 형법에 대한 객관식 질문입니다. 그리고 너는 전문 변호사로서 아래 과정을 수행하고 최대한 정확한 답을 내려야해.
      아래의 예시의 답변과 같은 형식으로 답변을 생성해.
      정확한 답을 하기 위해 반드시 웹 브라우징을 활용하시오. 먼저 자세한 정답 추론/해설 과정을 한글로 생성하세요.
      각 항목마다 reference에 관련 내용이 있는지 확인하세요. 그리고 각 항목에 대한 설명이 옳은지 판단하세요.
      그리고나서, 최종 답변은 반드시 다음과 같은 포맷으로 답해야 합니다. ’따라서, 정답은 (A|B|C|D)입니다.’

      ===========
      예시:

      reference:
      형법의 적용범위, 제1조(범죄의 성립과 처벌)  
      ① 범죄의 성립과 처벌은 행위 시의 법률에 따른다.
      ② 범죄 후 법률이 변경되어 그 행위가 범죄를 구성하지 아니하게 되거나 형이 구법(舊法)보다 가벼워진 경우에는 신법(新法)에 따른다.
      ③ 재판이 확정된 후 법률이 변경되어 그 행위가 범죄를 구성하지 아니하게 된 경우에는 형의 집행을 면제한다.

      제253조(시효의 정지와 효력) 
      ①시효는 공소의 제기로 진행이 정지되고 공소기각 또는 관할위반의 재판이 확정된 때로부터 진행한다.
      ②공범의 1인에 대한 전항의 시효정지는 다른 공범자에게 대하여 효력이 미치고 당해 사건의 재판이 확정된 때로부터 진행한다.
      ③범인이 형사처분을 면할 목적으로 국외에 있는 경우 그 기간 동안 공소시효는 정지된다.
      ④ 피고인이 형사처분을 면할 목적으로 국외에 있는 경우 그 기간 동안 제249조제2항에 따른 기간의 진행은 정지된다.


      질문: 공소시효에 대한 설명으로 옳지 않은 것은?
      선택지:
      (A). 범죄 후 법률의 개정에 의하여 법정형이 가 벼워진 경우에는 형법 제1조 제2항에 의하여 당해 범죄 사실에 적용될 가벼운 법정형인 신법의 법정형이 공소시효 기간의 기준이 된다.
      (B). 1개의 행위가 형법 상 사기죄와 변호사법 위반죄에 해당하고 양 죄가 상상적 경합 관계에 있는 경우, 변호사법 위반죄의 공소시효가 완성되었다면 사기죄의 공소시효도 완성된 것으로 보아야 한다.
      (C). 공범의 1인으로 기소된 자가 범죄의 증명이 없다는 이유로 무죄의 확정 판결을 선 고 받은 경우, 그는 공범이라고 할 수 없으므로 그에 대하여 제기된 공소는 진범에 대한 공소시효를 정 지시키는 효력이 없다.
      (D). 공범의 1인에 대한 공소시효 정지는 다른 공범자에게 대하여 그 효력이 미치는데 , 여기의 ‘공범’에는 뇌물공여죄와 뇌물수수죄 사이와 같은 대 향범 관계에 있는 자는 포함되지 않는다.

      차근 차근 생각해보겠습니다

      답변:
      reference가 주어졌습니다.
      thinking_process:
      (A).reference에 관련 내용이 있습니다. 형법 제1조 제2항은 범죄 후 법률이 변경되어 그 행위가 범죄를 구성하지 않게 되거나 형이 구법보다 가벼워진 경우, 신법을 적용하도록 규정하고 있습니다. 따라서, 법정형이 가벼워진 경우에는 신법의 법정형이 공소시효 기간의 기준이 됩니다.

      (B). reference에 관련 내용이 없습니다. 다만 제가 아는 지식을 최대한 정확하게 살펴보면 상상적 경합 관계에서는 여러 범죄가 하나의 행위로 처벌되므로, 공소시효는 가장 중한 법정형을 기준으로 진행됩니다. 따라서, 변호사법 위반죄의 공소시효가 완성되었더라도 사기죄의 공소시효는 여전히 진행 중일 수 있습니다. 이러한 경우, 사기죄의 공소시효가 완성되었는지 여부는 별도로 판단해야 합니다.
    
      (C). reference에 관련 내용이 있습니다. 다만 내용이 조금 부족하여 저의 지식과 함께 답변하면 형사소송법 제253조 제2항에 따르면, 공범의 1인에 대한 공소제기로 공소시효가 정지되며, 이는 다른 공범자에게도 효력이 미칩니다. 따라서, 무죄 판결을 받은 공범이 있더라도 그에 대한 공소제기가 다른 공범자의 공소시효에 영향을 미칩니다.
      
      (D). reference에 관련 내용이 있습니다. 다만 내용이 조금 부족하여 저의 지식과 함께 답변하면 형사소송법 제253조 제2항은 공범의 1인에 대한 공소제기로 공소시효가 정지되며, 이는 다른 공범자에게도 효력이 미친다고 규정하고 있습니다. 여기서의 '공범'에는 대향범 관계에 있는 자도 포함됩니다. 따라서, 뇌물공여죄와 뇌물수수죄 사이의 대향범 관계에 있는 자도 공범으로 간주되어 공소시효 정지의 효력이 미칩니다.

      final_answer:
        따라서, 옳지 않은 설명은 (B)입니다.
        따라서, 정답은 (B)입니다.
      ===========

      reference:
      {context}

      질문: {question}
      
      선택지:
      (A). {A}
      (B). {B}
      (C). {C}
      (D). {D}
      """

    system_prompt = SystemMessagePromptTemplate.from_template(
        template=system_message_prompt
    )
    return system_prompt

  def __get_ai_prompt_template(self) -> AIMessagePromptTemplate:
    ai_prompt = AIMessagePromptTemplate.from_template(
      template = "차근 차근 생각해보겠습니다"
    )
    return ai_prompt
  
  def get_creaet_search_termrompt(self) -> ChatPromptTemplate:
    system_message_prompt = """
      주어진 문제와 선지를 풀어내기 위한 검색어(search_term)를 생성해야합니다.
      검색의 특성을 고려하여 가장 좋은 결과가 기대되는 검색어(search_term)를 생성해주세요.
      검색어(search_term)는 단어일수도 있고 문장일 수도 있지만 전체 내용이 포함되어야합니다. question보다는 choice에 더 가까운 검색어를 생성해주세요.

      ===========
      예시:
      문제: 공소장 변경에 대한 설명으로 옳지 않은 것은?
      선지: 법원은 검사가 공소장 변경을 신청한 경우 피고인이나 변호인의 청구가 있는 때에는 피고인으로 하여 금 필요한 방어의 준비를 하게 하기 위해 필요한 기간 공판 절차를 정지하여야 한다.

      search_term:
      "공소장 변경 법원 피고인 방어 준비 공판 절차 정지 의무"
      ===========

      문제: {question}
      선지: {choice}
      """
    system_prompt = SystemMessagePromptTemplate.from_template(
        template=system_message_prompt
    )
    return ChatPromptTemplate.from_messages([
      system_prompt,
      ])
    