# Embedding Model Implementation for a Company Interview Coding Assignment

#### Notes:

- Overall, it took me ~6 hours to implement this assignment.
- For this assignment, I used [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- The average model performance speed is ~250 tokens/second.
- The model accurately interpets NLP-based single-column queries.

#### If I had more time...

Several experimentation options here:

- Instead of using a pre-trained embedding model, I would fine-tune my own on an artificial dataset generated by ChatGPT and/or Claude.
- I would take a small open-source LLM model (like [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B)) and fine tune it to perform semantic parsing. For example, `apparel product with a distance shorter than 200 and a high priority` would be parsed into structured elements:
  ```
  [
      {column: "Product_Category", value: "Apparel"},
      {column: "Delivery_distance", operator: "<", value: 200},
      {column: "Priority", value: "High"}
  ]
  ```
  The quantized Llama 3.2 1B models have demonstrated the capability to achieve over 350 tokens per second in the prefill phase and over 40 tokens per second in the decode stage on mobile devices.
