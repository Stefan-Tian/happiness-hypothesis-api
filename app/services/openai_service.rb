require 'openai'
require 'logger'
require 'matrix'
require 'polars-df'

class OpenaiService
  CHAT_COMPLETION_MODEL = "gpt-3.5-turbo"
  EMBEDDING_MODEL = "text-embedding-ada-002"
  SEPARATOR = "\n* "
  SEPARATOR_LEN = 4
  MAX_SECTION_LENGTH = 1000

  def initialize
    @client = OpenAI::Client.new
    setup_logger
  end

  def setup_logger
    @logger = Logger.new($stdout)
    @logger.formatter = proc do |severity, datetime, progname, msg|
      "#{datetime} [#{severity}] #{msg}\n"
    end
  end

  def load_embeddings(fname)
    @logger.info("Loading embeddings from #{fname}")

    page_df = Polars.read_csv(fname)
    max_dim = page_df.columns.reject { |column| column == "title" }.map(&:to_i).max
    embeddings = {}
    page_df.each_row do |row|
      title = row['title']
      values = (0..max_dim).map { |i| row[i.to_s] }
      embeddings[title] = values
    end

    @logger.info("Embeddings loaded successfully")
    puts embeddings
    embeddings
  rescue StandardError => e
    @logger.error("Error loading embeddings: #{e.message}")
    nil
  end

  def answer_query_with_context(query, page_df, context_embeddings)
    @logger.info("Generating answer for query: #{query}")
    prompt, context = construct_prompt(query, page_df, context_embeddings)

    response = @client.chat(
      parameters: {
        model: CHAT_COMPLETION_MODEL,
        messages: [{ role: "user", content: prompt }]
      }
    )

    answer = response.dig("choices", 0, "message", "content")
    return answer, context
  rescue StandardError => e
    @logger.error("An error occurred while generating the answer: #{e.message}")
    return nil
  end

  def construct_prompt(question, page_df, context_embeddings)
    most_relevant_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections = construct_chosen_sections(most_relevant_sections, page_df)

    header = construct_header
    questions_and_answers = construct_predefined_questions_and_answers

    return header + chosen_sections.join + questions_and_answers + "\n\n\nQ: #{question}\n\nA:", chosen_sections.join
  end

  def construct_chosen_sections(most_relevant_sections, page_df)
    chosen_sections = []
    chosen_section_token_len = 0

    most_relevant_sections.each do |similarity, doc_index|
      section = page_df[page_df["title"] == doc_index]
      next if section.empty?
      chosen_section_token_len += section["tokens"][0] + SEPARATOR_LEN

      content_to_add = if chosen_section_token_len > MAX_SECTION_LENGTH
        space_left = MAX_SECTION_LENGTH - chosen_section_token_len - SEPARATOR_LEN
        section["content"][0..space_left]
      else
        section["content"]
      end

      chosen_sections << SEPARATOR + content_to_add.to_a.join
    end

    chosen_sections
  end

  def construct_header
    "Jonathan Haidt is a social psychologist and author of 'The Happiness Hypothesis.' Below are questions and answers by him. Responses are concise, limited to three sentences, and complete. The dialogue stops immediately once the point is made.\n\nContext for better understanding is derived from 'The Happiness Hypothesis':\n"
  end

  def construct_predefined_questions_and_answers
    question_1 = "\n\n\nQ: What inspired you to explore the concept of happiness?\n\nA: I was intrigued by how ancient wisdom and modern psychology both consider happiness a significant pursuit. Observing commonalities in happiness across various cultures and ages led me to explore this further. My journey started with curiosity, evolving into an extensive study."
    question_2 = "\n\n\nQ: How does the 'elephant and the rider' metaphor serve in understanding happiness?\n\nA: The metaphor illustrates the relationship between our emotional and rational sides. The rider, representing rationality, struggles to guide the elephant, our emotional side. This struggle highlights the complexities of steering our happiness due to underlying instincts and learned behaviors."
    question_3 = "\n\n\nQ: Can money buy happiness?\n\nA: Money contributes to comfort, which can alleviate stress, but it's not a direct path to happiness. True contentment often comes from relationships, meaningful work, and personal growth. There's a threshold where more money doesn't equal more happiness."
    question_4 = "\n\n\nQ: How do modern societies complicate our pursuit of happiness?\n\nA: Modern societies often emphasize material success and competition, leading to a 'rat race' which can overshadow true happiness. Social comparison exacerbated by social media also contributes to dissatisfaction. Our pursuit of external validation often conflicts with internal contentment."
    question_5 = "\n\n\nQ: What role do adversity and suffering play in achieving happiness?\n\nA: Adversity introduces essential growth and resilience, teaching us to appreciate joy even more. Suffering makes us more empathetic and understanding, deepening our connections with others. Essentially, it’s not the absence of suffering but how we respond to it that shapes our happiness."
    question_6 = "\n\n\nQ: How does one's moral foundation influence their happiness?\n\nA: Our moral foundation guides our actions, affecting our social relationships and sense of personal integrity. Aligning actions with our moral compass tends to foster a sense of purpose and satisfaction. It’s a social and psychological anchor, directly influencing our subjective well-being."
    question_7 = "\n\n\nQ: Why did you choose 'The Happiness Hypothesis' as your book title?\n\nA: The title reflects the exploration of happiness as both ancient wisdom and a modern psychological construct. It signifies an inquiry into various 'hypotheses' of happiness that humanity has held throughout history. The goal was to synthesize these perspectives, identifying core truths about human flourishing."
    question_8 = "\n\n\nQ: How long did it take to write 'The Happiness Hypothesis'?\n\nA: It took several years of research, compiling and reflecting upon diverse psychological studies and historical texts. The writing itself was an iterative process, spanning over a couple of years. This journey was as much about my understanding as it was about conveying the concept."
    question_9 = "\n\n\nQ: What's the best approach to teaching happiness in academic settings?\n\nA: Incorporating it into curricula from early education, focusing on emotional intelligence, resilience, and mindfulness. For higher education, interdisciplinary courses that combine philosophy, psychology, and real-life applications are effective. It’s crucial to move beyond theory to practices that enhance students' well-being."
    question_10 = "\n\n\nQ: How do you know when to end the pursuit of a certain path to happiness?\n\nA: When the pursuit itself becomes a source of distress or leads to a neglect of personal relationships and health, it's a sign. If the path fosters negative behaviors or is misaligned with your values, it's time to reassess. Recognizing that some paths are unfulfilling allows for the exploration of more authentic routes to happiness."

    [question_1, question_2, question_3, question_4, question_5, question_6, question_7, question_8, question_9, question_10].join
  end

  def order_document_sections_by_query_similarity(query, context_embeddings)
    query_embedding = get_query_embedding(query)

    similarities = context_embeddings.map do |doc_index, doc_embedding|
      similarity = vector_similarity(query_embedding, doc_embedding)
      [similarity, doc_index]
    end

    similarities.sort_by { |similarity, doc_index| similarity }.reverse
  end

  def vector_similarity(x, y)
    Vector.elements(x).inner_product(Vector.elements(y))
  end

  def get_query_embedding(query)
    @logger.info("Fetching embedding for query: #{query}")

    response = @client.embeddings(
      parameters: {
        model: EMBEDDING_MODEL,
        input: query
      }
    )

    embedding = response["data"].first["embedding"]
    @logger.info("Embedding fetched successfully")
    embedding
  rescue StandardError => e
    @logger.error("Error fetching query embedding: #{e.message}")
    nil
  end

  def process_question(question_asked)
    previous_question = Question.find_by(question: question_asked)

    if previous_question
      previous_question.increment!(:ask_count)
      return previous_question
    end

    page_df = Polars.read_csv("tmp/book_pages.csv")
    document_embeddings = load_embeddings('tmp/book_embeddings.csv')
    answer, context = answer_query_with_context(question_asked, page_df, document_embeddings)

    Question.create!(question: question_asked, answer: answer, context: context)
  end
end
