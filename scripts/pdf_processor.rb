#!/usr/bin/env ruby

require 'openai'
require 'dotenv'
require 'polars-df'
require 'pdf-reader'
require 'csv'
require 'logger'
require 'tiktoken_ruby'

# PdfProcessor is a class for processing PDF files to extract content,
# compute embeddings using OpenAI, and write the results to CSV files.
class PdfProcessor
  def initialize
    Dotenv.load if defined?(Dotenv) # Load .env file if dotenv is used

    OpenAI.configure do |config|
      config.access_token = ENV["OPENAI_API_KEY"]
    end

    @client = OpenAI::Client.new
    @model = "text-embedding-ada-002"
    @logger = Logger.new($stdout)
  end

  # Main method to perform the entire PDF processing workflow.
  # It includes reading the PDF, creating a dataframe, computing embeddings,
  # and writing the results to CSV files.
  # @param [String] pdf_file_path Path to the PDF file to be processed.
  def perform(pdf_file_path)
    @logger.info("Starting processing of #{pdf_file_path}")

    begin
      titles, contents, tokens = process_pdf(pdf_file_path)
      @logger.info("PDF processed successfully, creating dataframe...")

      df = create_dataframe(titles, contents, tokens)
      df.write_csv("tmp/book_pages.csv")
      @logger.info("Dataframe created and written to CSV successfully.")

      @logger.info("Computing embeddings...")
      book_embeddings = compute_embeddings(df)
      @logger.info("Embeddings computed successfully.")

      @logger.info("Writing embeddings to CSV...")
      write_embeddings_to_csv(book_embeddings)
      @logger.info("Embeddings written to CSV successfully.")
    rescue StandardError => e
      @logger.error("An error occurred: #{e.message}")
      @logger.debug(e.backtrace.join("\n"))  # For a detailed stack trace in the log
    end

    @logger.info("Processing completed.")
  end

  private

  # Processes the PDF file to extract titles, contents, and token counts of each page.
  # It handles unreadable files and logs the progress.
  # @param [String] pdf_file_path Path to the PDF file to be processed.
  # @return [Array<Array>] Arrays of titles, contents, and tokens.
  def process_pdf(pdf_file_path)
    reader = PDF::Reader.new(pdf_file_path)
    titles = []
    contents = []
    tokens = []

    # Check if the file exists and is readable
    unless File.exist?(pdf_file_path) && File.readable?(pdf_file_path)
      @logger.error("File does not exist or is not readable: #{pdf_file_path}")
      raise StandardError, "File does not exist or is not readable"
    end

    begin
      reader.pages.each_with_index do |page, index|
        content = page.text.split.join(" ")
        unless content.empty? || count_tokens(content) + 4 > 8191
          titles << "Page #{index + 1}"
          contents << content
          tokens << count_tokens(content) + 4  # Assuming the +4 is part of the token calculation logic
        end
      end
      @logger.info("Processed #{reader.page_count} pages.")

    rescue PDF::Reader::MalformedPDFError => e
      @logger.error("Malformed PDF file: #{e.message}")
      raise
    rescue StandardError => e
      @logger.error("Error processing PDF file: #{e.message}")
      raise
    end

    [titles, contents, tokens]
  end

  # Creates a dataframe from the extracted titles, contents, and tokens.
  # This dataframe is later used for computing embeddings and writing to CSV.
  # @param [Array<String>] titles Array of titles (page numbers).
  # @param [Array<String>] contents Array of page contents.
  # @param [Array<Integer>] tokens Array of token counts.
  # @return [Polars::DataFrame] The created dataframe.
  def create_dataframe(titles, contents, tokens)
    Polars::DataFrame.new([
      Polars::Series.new("title", titles),
      Polars::Series.new("content", contents),
      Polars::Series.new("tokens", tokens)
    ])
  end

  # Computes embeddings for each content in the dataframe.
  # This method logs the start, progress, and completion of the computation.
  # @param [Polars::DataFrame] df The dataframe with page contents.
  # @return [Hash] The computed embeddings indexed by page number.
  def compute_embeddings(df)
    embeddings = {}
    contents = df["content"].to_a

    @logger.info("Starting embeddings computation for #{contents.size} contents.")
    start_time = Time.now

    contents.each_with_index do |content, index|
      @logger.info("Computing embedding for content #{index + 1} of #{contents.size}")
      embeddings[index] = get_embeddings(content)
    end

    end_time = Time.now
    @logger.info("Embeddings computation completed in #{end_time - start_time} seconds.")

    embeddings
  end

  # Writes computed embeddings to a CSV file.
  # The CSV file contains embeddings for each page of the PDF.
  # @param [Hash] book_embeddings Hash of embeddings indexed by page number.
  def write_embeddings_to_csv(book_embeddings)
    CSV.open('tmp/book_embeddings.csv', 'wb') do |csv|
      # length of the output embedding vector is 1536 for the text-embedding-ada-002 model
      csv << ['title'] + (0..1535).to_a

      book_embeddings.each do |(i, embedding)|
        csv << ["Page #{i + 1}"] + embedding
      end
    end
  end

  # Counts the number of tokens in a given text.
  # This is used to ensure embedding computation stays within limits.
  # @param [String] text The text to be tokenized.
  # @return [Integer] The number of tokens in the text.
  def count_tokens(text)
    enc = Tiktoken.encoding_for_model(@model)
    enc.encode(text).length
  end

  # Retrieves embeddings for a given text using the OpenAI API.
  # Handles and logs errors in case of API failures.
  # @param [String] The text for which to retrieve embeddings.
  # @return [Array] The embedding vector for the text.
  def get_embeddings(text)
    response = @client.embeddings(
      parameters: {
        model: @model,
        input: text
      }
    )

    response['data'][0]['embedding']

  rescue OpenAI::Error => e
    @logger.error("An error occurred while retrieving embeddings: #{e.message}")
    raise
  rescue StandardError => e
    @logger.error("An unexpected error occurred while retrieving embeddings: #{e.message}")
    raise
  end
end

# Main execution logic
if ARGV.length != 1
  puts "Usage: #{$PROGRAM_NAME} <pdf_file_path>"
  exit
end

pdf_processor = PdfProcessor.new
pdf_processor.perform(ARGV[0])
