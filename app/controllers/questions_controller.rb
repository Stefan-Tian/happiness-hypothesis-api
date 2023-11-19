class QuestionsController < ApplicationController
  def ask
    question_asked = params[:question].to_s
    question_asked += '?' unless question_asked.end_with?('?')

    openai_service = OpenaiService.new
    question = openai_service.process_question(question_asked)

    if question.persisted?
      render json: { question: question.question, answer: question.answer, id: question.id }
    else
      render json: { question: question_asked, answer: nil }, status: :not_found
    end
  rescue => e
    Rails.logger.error("An error occurred: #{e.message}")
    render json: { error: e.message }, status: :internal_server_error
  end
end
