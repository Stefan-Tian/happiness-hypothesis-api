class Question < ApplicationRecord
  # Validations
  validates :question, presence: true
  validates :ask_count, numericality: { greater_than_or_equal_to: 0 }
end
