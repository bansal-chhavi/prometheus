// 1. Valid faithful answer

{
  "query": "What are the main benefits of regular exercise?",
  "rag_answer": "Regular exercise improves cardiovascular health, supports weight management, and boosts mood by releasing endorphins.",
  "sources": [
    "Regular physical activity strengthens the heart and lungs and improves circulation.",
    "Exercise helps maintain a healthy weight and reduces the risk of chronic disease.",
    "Physical activity releases endorphins, which can improve mood and reduce stress."
  ]
}

// 2. Hallucinated answer
{
  "query": "What does vitamin D do for the body?",
  "rag_answer": "Vitamin D directly cures anxiety and completely prevents all infections when taken daily.",
  "sources": [
    "Vitamin D helps the body absorb calcium and supports bone health.",
    "A deficiency in vitamin D can lead to weakened bones."
  ]
}

// 3. Partially supported / drift answer
{
  "query": "What are the effects of caffeine?",
  "rag_answer": "Caffeine increases alertness, can cause long-term heart damage, and improves digestion.",
  "sources": [
    "Caffeine can improve alertness and reduce fatigue.",
    "High caffeine intake may increase heart rate and blood pressure."
  ]
}